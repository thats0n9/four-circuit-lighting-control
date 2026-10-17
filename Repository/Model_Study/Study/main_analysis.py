"""Numerical model and evaluation routines."""
from pathlib import Path
import argparse,json,hashlib,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import analysis_core as c
from analysis_engine import fast_engine_trace
s=c.s
SEEDS=list(range(20261340,20261348))

def layers(tele,inp,z,cal,guard,ref):
    # Hierarchical causes for guarded infeasible automatic requests; not a
    # causal allocation of observed energy or of all reading deficits.
    req,on,off,dwell,healthy,db,node=tele.T
    masked=req&~off
    ds_model=c.deficit(cal['pred'],cal['dayhat'],s.m.ZONES,c.EDGES[3],ref['margin_lx'])
    ds_true=c.deficit(z['true_H'],z['true_day'],s.m.ZONES,c.EDGES[3],ref['margin_lx'])
    ds_guard=c.deficit(cal['pred'],cal['dayhat'],s.m.ZONES,c.EDGES[3],ref['margin_lx'],guard)
    _,_,ff,uf=c.lookups(ds_guard)
    model=np.zeros((3,16),bool);truth=np.zeros_like(model)
    for rq in range(16):
        ix=s.m.STATES[rq].astype(bool)
        model[:,rq]=np.any(np.all(ds_model[:,:,ix]<=1e-12,axis=2),axis=1) if rq else True
        truth[:,rq]=np.any(np.all(ds_true[:,:,ix]<=1e-12,axis=2),axis=1) if rq else True
    # Instantaneous true feasibility on the SAME 81-point planning grid.
    D=inp['window'];instant=np.ones((s.m.SECONDS,16),bool)
    zonegood=np.zeros((s.m.SECONDS,16,4),bool)
    for zone in range(4):
        ix=s.m.ZONES==zone;lamps=s.m.STATES@z['true_H'][ix].T
        for state in range(16):
            E=lamps[state]+D[:,None]*z['true_day'][ix]
            av=E.mean(1);mi=E.min(1)
            zonegood[:,state,zone]=(av>=500)&(mi>=300)&(mi/np.maximum(av,1)>=.6)
    for rq in range(1,16):instant[:,rq]=zonegood[:,:,s.m.STATES[rq].astype(bool)].all(2).any(1)
    f_instant=instant[np.arange(len(tele))//5,masked]
    candidates=healthy.astype(bool)&(masked!=0)
    infeasible=candidates&~ff[db,masked,on,off]
    physical=infeasible&~f_instant
    binning=infeasible&~physical&~truth[db,masked]
    modelcause=infeasible&~physical&~binning&~model[db,masked]
    protection=infeasible&~physical&~binning&~modelcause&~uf[db,masked]
    constraints=infeasible&~physical&~binning&~modelcause&~protection
    assert np.array_equal(physical|binning|modelcause|protection|constraints,infeasible)
    constrained_feasible=candidates&ff[db,masked,on,off]
    current_good=ds_guard[db,node]
    current_req_bad=(current_good*s.m.STATES[masked]).max(1)>1e-12
    return dict(automatic_request_frames=int(candidates.sum()),guarded_infeasible_frames=int(infeasible.sum()),
        physical_frames=int(physical.sum()),day_interval_frames=int(binning.sum()),model_frames=int(modelcause.sum()),
        protection_frames=int(protection.sum()),manual_or_hold_frames=int(constraints.sum()),
        dwell_blocked_feasible_frames=int((constrained_feasible&current_req_bad&~dwell.astype(bool)).sum()),
        guarded_unrestricted_feasible_entries=int(uf[:,1:].sum()),controlled_entries=45)

def job(case):
    cid=case['case'];z,pool,field=c.load_case(case)
    ref=json.loads((s.R/(cid+'_reference_calibration.json')).read_text('utf-8'))
    np.savez_compressed(c.OUT/(cid+'_final_validation.npz'),points=z['final_points'],zones=z['final_zones'],
         true_H=z['final_H'],true_day=z['final_day'],prior=pool['final_prior'],direct_day=pool['final_direct_day'],
         validation_readings=pool['validation'],provenance=c.PROV)
    obs=c.make_protection(case,field,z)
    fits={};looks={};cov=[];params=[];scan=[];metriccov=[];metricparams=[]
    for layout in s.LAYOUTS:
        for n in s.COUNTS:
            cal=s.fit(z,pool,s.selections()[layout,n],'improved');cal['camera_margin']=ref['margin_lx']
            guard=c.protect(field,cal,obs)
            guard['metric']=c.metric_guard(cal,obs,c.EDGES[3],ref['margin_lx'])
            fits[layout,n]=(cal,guard)
            key=f'{layout}_{n}'
            for policy,uniform in [('P',False),('PU',True)]:
                ds=c.deficit(cal['pred'],cal['dayhat'],s.m.ZONES,c.EDGES[3],ref['margin_lx'],guard,uniform)
                lk,dz,ff,uf=c.lookups(ds);looks[key+'_'+policy]=lk
                if policy=='PU':looks[key+'_DZU']=dz
                for db in range(3):
                    for req in range(1,16):
                        needed=s.m.STATES[req].astype(bool)
                        scan.append(dict(provenance=c.PROV,case=cid,layout=layout,n_points=n,policy=policy,
                            day_bin=db,request_code=req,feasible_states=int((ds[db,:,needed]<=1e-12).all(0).sum())))
            cov+=c.coverage(z,cal,guard,layout,n,cid)
            metriccov+=c.metric_coverage(z,cal,guard,c.EDGES[3],ref['margin_lx'],layout,n,cid)
            for db in range(3):
                for state in range(16):
                    for zone in range(4):
                        metricparams.append(dict(provenance=c.PROV,case=cid,layout=layout,n_points=n,day_bin=db,state=state,zone=zone+1,
                            mean_positive_bound_lx=float(guard['metric'][db,state,zone,0]),
                            minimum_positive_bound_lx=float(guard['metric'][db,state,zone,1]),
                            uniformity_positive_bound=float(guard['metric'][db,state,zone,2])))
            for state in range(16):
                for zone in range(4):
                    params.append(dict(provenance=c.PROV,case=cid,layout=layout,n_points=n,state=state,zone=zone+1,
                      night_positive_bound_lx=float(guard['nplus'][state,zone]),night_negative_bound_lx=float(guard['nminus'][state,zone]),
                      day_positive_bound_per_lx=float(guard['dplus'][zone]),day_negative_bound_per_lx=float(guard['dminus'][zone])))
    # All Section S8 comparator variants use the same state-compensated reference.
    cal,g=fits['depth',12]
    for bins in [3,6,12]:
        plan=s.d.planner(cal['pred'],cal['dayhat'],s.m.ZONES,c.EDGES[bins],cal['margin'],cal['day_margin'],ref['margin_lx'])
        looks['legacyPU'+str(bins)]=plan['lookup'];looks['legacyDU'+str(bins)]=plan['union']
        gb=dict(g);gb['metric']=c.metric_guard(cal,obs,c.EDGES[bins],ref['margin_lx'])
        ds=c.deficit(cal['pred'],cal['dayhat'],s.m.ZONES,c.EDGES[bins],ref['margin_lx'],gb)
        lk,dz,_,_=c.lookups(ds);looks['currentPU'+str(bins)]=lk;looks['currentDU'+str(bins)]=dz
    # Uniform-margin comparator recalculated with the same study reference envelope.
    assert np.array_equal(looks['currentPU3'],looks['depth_12_PU'])
    np.savez_compressed(c.OUT/(cid+'_lookups.npz'),provenance=c.PROV,**looks)
    raw=[];contrasts=[];layerrows=[];s8=[];traces={}
    for day,seed in enumerate(SEEDS,1):
        inp=s.m.construct(seed);table=s.r.bins_table(inp,ref,c.EDGES[3]);manual=np.array(inp['manuals'],np.int64).reshape(-1,4)
        qt=c.quality_tables(inp,z);cache={}
        baseids,vi,bb,bt=fast_engine_trace(inp['det'],inp['valid'],table,looks['depth_12_PU'],manual)
        # Passive telemetry must not change executable behaviour.
        if day==1:
            plain=s.fast_engine(inp['det'],inp['valid'],table,looks['depth_12_PU'],manual)
            assert np.array_equal(plain[0],baseids) and np.array_equal(plain[2],bb) and plain[1]==vi
        base_result=c.evaluate(baseids,inp,z,qt,vi)
        auto=qt[-1];eligible=np.repeat(auto.any(1),5)&bt[:,4].astype(bool)&(bt[:,1]==0)&(bt[:,2]==0)
        ccv=np.count_nonzero(eligible)
        if day in [1,5]:layerrows.append(dict(provenance=c.PROV,case=cid,day=day,**layers(bt,inp,z,cal,g,ref)))
        all_outputs={}
        for layout in s.LAYOUTS:
            for n in s.COUNTS:
                for policy in ['P','PU','DZU']:
                    key=f'{layout}_{n}_{policy}';look=looks[key];h=hashlib.sha256(look.tobytes()).hexdigest()
                    if h not in cache:
                        ids,viol,bins=s.fast_engine(inp['det'],inp['valid'],table,look,manual)
                        cache[h]=(ids,c.evaluate(ids,inp,z,qt,viol))
                    ids,res=cache[h]
                    all_outputs[key]=ids
                    row=dict(provenance=c.PROV,case=cid,layout=layout,n_points=n,policy=policy,day=day,input_seed=seed,**res)
                    raw.append(row)
                    contrasts.append(dict(provenance=c.PROV,case=cid,layout=layout,n_points=n,policy=policy,day=day,
                       energy_difference_kWh=res['total_incremental_kWh']-base_result['total_incremental_kWh'],
                       Jauto_difference_pp=res['Jauto_pct']-base_result['Jauto_pct'],
                       Fauto_difference_pp=res['Fauto_pct']-base_result['Fauto_pct'],
                       Uauto_difference_pp=res['Uauto_pct']-base_result['Uauto_pct'],
                       active_controllable_frames=ccv,
                       changed_active_controllable_frames=int(((ids!=baseids)&eligible).sum()),
                       active_changed_pct=100*int(((ids!=baseids)&eligible).sum())/max(ccv,1)))
                    if layout=='depth' and n in [12,24] and policy=='PU':traces[f'D{day:02}_{n}']=ids
        np.savez_compressed(c.OUT/(cid+f'_D{day:02}_all_policy_traces.npz'),dt_s=.2,input_seed=seed,provenance=c.PROV,**all_outputs)
        # Separate current/legacy protection, same camera and same fresh inputs.
        for bins in [3,6,12]:
            tab=s.r.bins_table(inp,ref,c.EDGES[bins])
            for family in ['legacy','current']:
                for policy in ['PU','DU']:
                    ids,viol,used=s.fast_engine(inp['det'],inp['valid'],tab,looks[family+policy+str(bins)],manual)
                    s8.append(dict(provenance=c.PROV,case=cid,day=day,input_seed=seed,family=family,
                         bins=bins,policy=policy,camera_margin_lx=ref['margin_lx'],**c.evaluate(ids,inp,z,qt,viol)))
                    if family=='current' and policy=='PU' and bins==3:assert np.array_equal(ids,baseids)
        del qt,cache
    np.savez_compressed(c.OUT/(cid+'_reference_traces.npz'),provenance=c.PROV,dt_s=.2,**traces)
    spatial=[];noises=[]
    for layout in s.LAYOUTS:
        for n in s.COUNTS:
            for method in ['original','improved']:
                cc=s.fit(z,pool,s.selections()[layout,n],method)
                spatial.append(dict(provenance=c.PROV,case=cid,layout=layout,n_points=n,method=method,**s.spatial(z,pool,cc)))
    for rep in range(1,11):
        rng=np.random.default_rng(int(case['calibration_seed'])+1410000+rep*1000)
        nt=s.m.STATES@z['true_H'].T;nn=np.maximum(0,nt[None]+rng.normal(0,.008*nt[None]+.7,size=(4,16,81)))
        lev=np.array([100.,250.,500.,800.]);dd=lev[:,None]*z['true_day']
        dr=np.maximum(.1,dd[None]+rng.normal(0,.008*dd[None]+.7,size=(6,4,81)))
        rr=np.maximum(.1,lev[None]+rng.normal(0,.008*lev[None]+.7,size=(6,4)))
        for layout in s.LAYOUTS:
            for n in s.COUNTS:
                for method in ['original','improved']:
                    cc=s.fit(z,pool,s.selections()[layout,n],method,nn,dr,rr)
                    noises.append(dict(provenance=c.PROV,case=cid,layout=layout,n_points=n,method=method,replicate=rep,
                        RMSE_lx=s.spatial(z,pool,cc)['RMSE_lx']))
    return raw,contrasts,cov,params,scan,layerrows,s8,metriccov,metricparams,spatial,noises

def summarize(allrows):
    raw,con,cov,params,scan,layers_,s8,metriccov,metricparams,spatial,noises=allrows
    grouped=[];scene=[]
    for case in sorted({r['case'] for r in raw}):
        for layout in s.LAYOUTS:
            for n in s.COUNTS:
                for policy in ['P','PU','DZU']:
                    rr=[r for r in raw if (r['case'],r['layout'],r['n_points'],r['policy'])==(case,layout,n,policy)]
                    cr=[r for r in con if (r['case'],r['layout'],r['n_points'],r['policy'])==(case,layout,n,policy)]
                    cv=[r for r in cov if (r['case'],r['layout'],r['n_points'])==(case,layout,n)]
                    mv=[r for r in metriccov if (r['case'],r['layout'],r['n_points'])==(case,layout,n)]
                    accepted=sum(r['accepted_zone_state_levels'] for r in mv if r['metric']=='mean')
                    false_accepted=sum(r['false_accept_zone_state_levels'] for r in mv if r['metric']=='mean')
                    row=dict(provenance=c.PROV,case=case,layout=layout,n_points=n,policy=policy,
                        protection_min_zone_component_coverage_pct=min(r['two_sided_coverage_pct'] for r in cv),
                        metric_min_zone_bin_coverage_pct=min(r['one_sided_coverage_pct'] for r in mv),
                        metric_false_accept_zone_state_levels=false_accepted,
                        metric_accepted_zone_state_levels=accepted,
                        target_acceptance_pct=100*(1-false_accepted/max(accepted,1)) if accepted else 0.,
                        # Finite research qualification screen on the independent 208-point set: component coverage and
                        # zero observed target false acceptance. The metric
                        # buffers are NOT certified 95% lower confidence bounds.
                        protection_qualified=int(min(r['two_sided_coverage_pct'] for r in cv)>=95 and accepted>0 and false_accepted==0),
                        point_readings=88*n,point_labour_hours=88*n*50/3600)
                    for k in ['total_incremental_kWh','Fauto_pct','Uauto_pct','Jauto_pct','transitions']:
                        row[k]=float(np.mean([r[k] for r in rr]))
                    for k in ['energy_difference_kWh','Jauto_difference_pp','Fauto_difference_pp','Uauto_difference_pp','active_changed_pct']:
                        row[k]=float(np.mean([r[k] for r in cr]))
                    scene.append(row)
    for layout in s.LAYOUTS:
        for n in s.COUNTS:
            for policy in ['P','PU','DZU']:
                rr=[r for r in scene if (r['layout'],r['n_points'],r['policy'])==(layout,n,policy)]
                row=dict(provenance=c.PROV,layout=layout,n_points=n,policy=policy,
                    protection_qualified_scenarios=sum(r['protection_qualified'] for r in rr),
                    point_labour_hours=88*n*50/3600)
                for k in ['total_incremental_kWh','Fauto_pct','Uauto_pct','Jauto_pct','energy_difference_kWh','Jauto_difference_pp','active_changed_pct']:
                    for stat,v in c.describe([r[k] for r in rr]).items():row[k+'_'+stat]=v
                grouped.append(row)
    c.write('current_scene_summary.csv',scene);c.write('current_design_summary.csv',grouped)
    ss=[]
    for family in ['legacy','current']:
        for bins in [3,6,12]:
            for policy in ['PU','DU']:
                rr=[r for r in s8 if (r['family'],r['bins'],r['policy'])==(family,bins,policy)]
                ss.append(dict(provenance=c.PROV,family=family,bins=bins,policy=policy,
                    records=len(rr),energy_kWh_day=float(np.mean([r['total_incremental_kWh'] for r in rr])),
                    Jauto_pct=float(np.mean([r['Jauto_pct'] for r in rr])),Uauto_pct=float(np.mean([r['Uauto_pct'] for r in rr]))))
    c.write('S8_current_reference_summary.csv',ss)
    c.save('summary.json',dict(provenance=c.PROV,protocol=c.read(c.OUT/'protocol_counts.csv'),
         current_daily_records=len(raw),coverage_records=len(cov),controlled_scan_records=len(scan),
         S8_daily_records=len(s8),current_design=grouped,S8_summary=ss,
         strict_depth12_feasible_entries=sum(r['feasible_states']>0 for r in scan if (r['layout'],r['n_points'],r['policy'])==('depth',12,'PU')),
         legacy_depth12_feasible_entries=0,
         manual_conflicts=sum(r['manual_conflicts'] for r in raw),
         common_input_seeds=SEEDS,quality_dt_s=.2,weather_hold_s=1.,reference_dt_s=1.,
         inferential_scope='36 parameter combinations and 8 common new days, not 36 independent buildings',
         primary_endpoint='Jauto: mean/minimum/uniformity union on the same non-manual-off reading-zone denominator',
         engineering_scales={'Jauto_pp':.2,'energy_kWh_day':.05,'active_action_pct':1.,'coverage_pct':95.},
         thresholds_role='declared decision scales, not universal standards or significance tests',
         protection_acceptance='each component >=95% finite-point coverage; at least one accepted strict state; zero observed target false acceptance',
         metric_buffer_scope='one-sided planning buffers, not certified 95% lower confidence bounds; full-range numerical coverage also reported'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--cases',nargs='*');args=ap.parse_args()
    c.OUT.mkdir(exist_ok=True)
    cases=c.read(s.F/'room_scenarios.csv')
    if args.cases:cases=[x for x in cases if x['case'] in args.cases]
    counts=[dict(item='new_common_input_days',count=8),dict(item='room_parameter_combinations',count=36),
       dict(item='original_spatial_fits',count=864),dict(item='current_improved_configurations',count=432),
       dict(item='protection_positions_per_case',count=69),dict(item='terminal_positions_per_case',count=208),
       dict(item='protection_night_readings_per_case',count=2*16*69),
       dict(item='protection_daylight_readings_per_case',count=2*4*69)]
    c.write('protocol_counts.csv',counts)
    for seed in SEEDS:
        inp=s.m.construct(seed)
        np.savez_compressed(c.OUT/(f'input_{seed}.npz'),provenance=c.PROV,seed=seed,
            **{k:inp[k] for k in ['truth','work','presentation','det','valid','window','refgray']},
            manuals=np.array(inp['manuals'],np.int64),schedule=np.array(inp['schedule'],np.int64))
    totals=[[] for _ in range(11)];start=time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for f in as_completed([ex.submit(job,case) for case in cases]):
            result=f.result()
            for arr,x in zip(totals,result):arr.extend(x)
            print('completed',len(totals[0])//288,'/',len(cases),'elapsed',round(time.time()-start),flush=True)
    for rows in totals:rows.sort(key=lambda r:tuple(str(r.get(k,'')) for k in ['case','layout','n_points','policy','family','bins','day','zone','component','state']))
    for name,rows in zip(['current_control_daily.csv','current_contrasts_daily.csv','protection_terminal_coverage.csv',
        'protection_parameters.csv','current_feasibility_scan.csv','feasibility_runtime_layers.csv','S8_current_reference_daily.csv',
        'protection_metric_terminal_coverage.csv','protection_metric_parameters.csv',
        'spatial_metrics.csv','spatial_noise_replicates.csv'],totals):c.write(name,rows)
    summarize(totals);print('Main analysis completed',round(time.time()-start,2),flush=True)

if __name__=='__main__':main()
