"""Numerical model and evaluation routines."""
from pathlib import Path
import csv,json,shutil,hashlib,math
import numpy as np
import analysis_core as c
from analysis_engine import fast_engine_trace
W=Path(__file__).resolve().parent;BASE=W/'analysis_results';OUT=W/'analysis_results'
PROV=''
def write(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=keys);wr.writeheader();wr.writerows(rows)
def save(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def automatic_reading(inp):
    auto=inp['work'].astype(bool).copy()
    for t,z,value,duration in inp['manuals']:
        if value==0:auto[t//5:min(len(auto),(t+duration)//5),z]=False
    return auto
def qualified_table(inp,ref,edges):
    D=inp['window'];noise=inp['refgray']-c.s.m.gfunc(D)
    gray=np.clip(c.s.p.camera_forward(D[:,None]+1.04*np.array(ref['true_reference_lamp_lx'])[None])+noise[:,None],0,255)
    corrected=np.interp(gray,ref['gray_knots'],ref['lux_knots'])-np.array(ref['lamp_camera_equivalent_lx'])[None]
    invalid=(gray<ref['gray_knots'][0]-2)|(gray>ref['gray_knots'][-1])|~np.isfinite(gray)
    # Last bin is qualified only for the declared physical D <=1260 lx domain.
    # This domain is an externally commissioned condition, not an estimate of D.
    bins=np.maximum(np.searchsorted(edges,np.maximum(0,corrected-ref['margin_lx']),side='right')-1,0).astype(np.int16)
    bins[invalid]=-1
    return bins,corrected,gray
def camera_qualification():
    rows=c.read(W/'control_calibration/reference_mixed_holdout.csv');detail=[];summ=[]
    for case in c.read(c.s.F/'room_scenarios.csv'):
        cid=case['case'];ref=json.loads((W/'camera_reference'/f'{cid}_reference_calibration.json').read_text('utf8'));M=ref['margin_lx']
        rr=[r for r in rows if r['case']==cid]
        for r in rr:
            gray=float(r['gray']);state=int(r['state']);D=float(r['true_day_lx'])
            raw=float(np.interp(gray,ref['gray_knots'],ref['lux_knots'])-ref['lamp_camera_equivalent_lx'][state]);e=raw-D
            valid=ref['gray_knots'][0]-2<=gray<=ref['gray_knots'][-1] and D<=1260
            X=max(0,raw-M)
            for nb,edges in c.EDGES.items():
                b=max(0,int(np.searchsorted(edges,X,side='right')-1));lo=float(edges[b]);hi=float(edges[b+1]+2*M) if b+1<len(edges) else 1260.
                detail.append(dict(case=cid,role=r['role'],session=r['session'],state=state,D_lx=D,bins=nb,
                    raw_estimate_lx=raw,residual_lx=e,margin_lx=M,qualified_input=int(valid),
                    interval_lower_lx=lo,interval_upper_lx=hi,contained=int(valid and lo-1e-9<=D<=hi+1e-9),
                    disposition='qualify' if valid else 'hold_automatic_and_recalibrate_range'))
        for role in ['margin','holdout_mixed']:
            r0=[r for r in detail if r['case']==cid and r['role']==role and r['bins']==3];rv=[r for r in r0 if r['qualified_input']]
            summ.append(dict(case=cid,role=role,margin_lx=M,records=len(r0),qualified=len(rv),excluded=len(r0)-len(rv),
                residual_min_lx=min(r['residual_lx'] for r in rv),residual_max_lx=max(r['residual_lx'] for r in rv),
                both_tails_within_margin=int(all(abs(r['residual_lx'])<=M for r in rv)),
                interval_contained=sum(r['contained'] for r in rv)))
    write('camera_interval_qualification.csv',detail);write('camera_interval_summary.csv',summ)
    assert all(r['both_tails_within_margin'] and r['qualified']==r['interval_contained'] for r in summ)
    # Supplement calibration-domain boundary tests, independent numerical probe.
    probes=[]
    for case in c.read(c.s.F/'room_scenarios.csv'):
        cid=case['case'];ref=json.loads((W/'camera_reference'/f'{cid}_reference_calibration.json').read_text('utf8'))
        for nb,edges in c.EDGES.items():
            levels=np.unique(np.r_[0,1260,np.clip(np.r_[edges-1e-6,edges,edges+1e-6]+ref['margin_lx'],0,1260)])
            for state in range(16):
                for D in levels:
                    total=float(D+1.04*ref['true_reference_lamp_lx'][state]);gray=float(c.s.p.camera_forward(total))
                    valid=gray<=ref['gray_knots'][-1] and gray>=ref['gray_knots'][0]-2
                    raw=float(np.interp(gray,ref['gray_knots'],ref['lux_knots'])-ref['lamp_camera_equivalent_lx'][state]);X=max(0,raw-ref['margin_lx'])
                    b=max(0,int(np.searchsorted(edges,X,side='right')-1));lo=float(edges[b]);hi=float(edges[b+1]+2*ref['margin_lx']) if b+1<nb else 1260
                    probes.append(dict(case=cid,bins=nb,state=state,D_lx=float(D),total_camera_lx=total,qualified_input=int(valid),
                        residual_lx=raw-D,contained=int(valid and lo<=D+1e-8 and D<=hi+1e-8)))
    write('camera_boundary_probes.csv',probes)
    assert all(r['contained'] for r in probes if r['qualified_input'])
    return summ
def replay_qualification():
    audit=[];inputs={d:c.s.m.construct(20261339+d) for d in range(1,9)}
    for case in c.read(c.s.F/'room_scenarios.csv'):
        cid=case['case'];ref=json.loads((W/'camera_reference'/f'{cid}_reference_calibration.json').read_text('utf8'));looks=np.load(BASE/(cid+'_lookups.npz'))
        for day,inp in inputs.items():
            manual=np.array(inp['manuals'],np.int64).reshape(-1,4);tr=np.load(BASE/(cid+f'_D{day:02}_all_policy_traces.npz'))
            for nb,edges in c.EDGES.items():
                table,raw,gray=qualified_table(inp,ref,edges);cache={}
                keys=[x for x in looks.files if x not in ['provenance'] and (x.startswith(('depth','area','window')) if nb==3 else False)]
                keys+= [f'{family}{pol}{nb}' for family in ['current','legacy'] for pol in ['PU','DU']]
                for key in keys:
                    lk=looks[key];sha=hashlib.sha256(lk.tobytes()).hexdigest()
                    if sha not in cache:cache[sha]=fast_engine_trace(inp['det'],inp['valid'],table,lk,manual)
                    ids,viol,bins,tele=cache[sha]
                    if key in tr.files:old=tr[key]
                    else:old=c.s.fast_engine(inp['det'],inp['valid'],c.s.r.bins_table(inp,ref,edges),lk,manual)[0]
                    sample=ids[::5];sec=np.arange(c.s.m.SECONDS);qualified=table[sec,sample]>=0
                    errors=raw[sec,sample]-inp['window'];sel=c.s.m.STATES[sample]
                    high=np.r_[edges[1:]+2*ref['margin_lx'],1260]
                    good=qualified&(inp['valid'][::5]>0)
                    cont=(inp['window']>=edges[np.maximum(bins,0)]-1e-8)&(inp['window']<=high[np.maximum(bins,0)]+1e-8)
                    audit.append(dict(case=cid,day=day,configuration=key,bins=nb,changed_frames=int(np.count_nonzero(ids!=old)),
                        range_invalid_seconds=int((~qualified).sum()),qualified_seconds=int(good.sum()),
                        contained_seconds=int((cont&good).sum()),max_absolute_residual_lx=float(abs(errors[good]).max()),manual_conflicts=int(viol)))
            print('qualified replay',cid,day,flush=True) if day==8 else None
    write('qualified_control_replay_audit.csv',audit)
    # A changed primary trace requires recomputing principal results rather than silent reuse.
    assert all(r['changed_frames']==0 for r in audit),'Primary outputs changed; rebuild study analysis with qualified engine'
    assert all(r['qualified_seconds']==r['contained_seconds'] for r in audit)
    return audit
def decision_examples():
    examples=[];full=[];codes=np.arange(16)
    cases=[('global_feasible',[[0 if s in [3,5,10,15] else .25]*4 for s in codes],15,0,0),
        ('single_zone_infeasible',[[.1 if s in [8,15] else .3,0 if s in [1,2,3] else .2,.1,.1] for s in codes],12,0,0),
        ('global_infeasible',[[.1+.01*(15-s)]*4 for s in codes],15,0,0),
        ('manual_locked',[[0 if s in [3,5,10,15] else .25]*4 for s in codes],14,8,1),
        ('multiple_feasible_supersets',[[0 if s in [1,7,11,15] else .2,0 if s in [2,7,11,15] else .2,0,0] for s in codes],12,0,0)]
    for name,vals,req,on,off in cases:
        ds=np.array(vals,float)[None];pu,dz,feas,_=c.lookups(ds);wanted=c.s.m.STATES[req].astype(bool)
        score=ds[0,:,wanted].max(0) if req else np.zeros(16)
        singles=[int(pu[0,b,0,0]) for b in c.BITS if req&b];union=int(np.bitwise_or.reduce(singles));union=(union|on)&~off
        examples.append(dict(example=name,request=req,forced_on=on,forced_off=off,single_zone_seeds='|'.join(map(str,singles)),
            constrained_union=union,PU=int(pu[0,req,on,off]),DZU=int(dz[0,req,on,off]),strict_feasible=int(feas[0,req,on,off])))
        for state in codes:full.append(dict(example=name,state=int(state),power_W=float(c.POWER[state]),
            zone1_deficit=vals[state][0],zone2_deficit=vals[state][1],zone3_deficit=vals[state][2],zone4_deficit=vals[state][3],
            requested_max_deficit=float(score[state]),allowed=int((state&on)==on and (state&off)==0)))
    write('decision_examples_summary.csv',examples);write('decision_examples_16_states.csv',full)
    # Equal zero-power light has U=0; the exact implementation also floors means <1 lx.
    zero=c.deficit(np.zeros((4,4)),np.zeros(4),np.arange(4),np.array([0.]),guard=None)
    assert np.all(zero==1)
    return examples
def paired_envelopes():
    from paired_quality_bounds import main as paired_main
    return paired_main()
def scene_and_input_maps():
    scenes=c.read(c.s.F/'room_scenarios.csv');dec=c.read(BASE/'commissioning_decision_screen.csv');metrics=c.read(BASE/'current_scene_summary.csv')
    feature=[]
    for case in scenes:
        cid=case['case'];r=next(x for x in dec if x['case']==cid);a=next(x for x in metrics if (x['case'],x['layout'],x['n_points'],x['policy'])==(cid,'depth','12','PU'))
        feature.append(dict(**{k:v for k,v in case.items() if k!='provenance'},baseline_Jauto_pct=float(a['Jauto_pct']),
            baseline_energy_kWh=float(a['total_incremental_kWh']),**{k:v for k,v in r.items() if k not in ['case','comparison']}))
    write('scenario_candidate_features.csv',feature)
    daily=[]
    for family,seeds in [('primary',range(20261340,20261348)),('independent',range(20261360,20261364))]:
        for day,seed in enumerate(seeds,1):
            inp=c.s.m.construct(seed);auto=automatic_reading(inp);D=inp['window']
            daily.append(dict(family=family,day=day,seed=seed,occupied_zone_hours=float(inp['truth'].sum()/3600),
                reading_zone_hours=float(inp['work'].sum()/3600),automatic_reading_zone_hours=float(auto.sum()/3600),
                daylight_peak_lx=float(D.max()),daylight_integral_klxh=float(D.sum()/3600000),
                invalid_input_seconds=float((inp['valid']==0).sum()/5),manual_events=len(inp['manuals']),
                projection_zone=int(inp['manuals'][0][1])+1,daylight_step_hold_s=1))
    write('input_day_exposure_summary.csv',daily)
    # One-day deletion is protocol sensitivity, not building-level inference.
    contrasts=c.read(BASE/'current_contrasts_daily.csv')
    loo=[]
    for r in dec:
        rr=[x for x in contrasts if (x['case'],x['layout'],x['n_points'],x['policy'])==(r['case'],'depth','24','PU')]
        for omitted in range(1,9):
            keep=[x for x in rr if int(x['day'])!=omitted];dj=float(np.mean([float(x['Jauto_difference_pp']) for x in keep]));de=float(np.mean([float(x['energy_difference_kWh']) for x in keep]))
            if r['both_protection_qualified']=='0':tag='protection_not_qualified'
            elif dj<=-.2 and de<=.05:tag='quality_gain_candidate'
            elif de<=-.05 and dj<=.2:tag='energy_gain_candidate_pending_meter_precision'
            elif dj>.2:tag='quality_worsens'
            else:tag='no_increment_above_declared_scales'
            loo.append(dict(case=r['case'],omitted_day=omitted,delta_J_pp=dj,delta_W_kWh=de,decision=tag,agrees_full=int(tag==r['decision'])))
    write('leave_one_input_day_out_screen.csv',loo)
    if not contrasts:
        names=[p.name for p in BASE.glob('*contrast*')];print('contrast filenames',names,flush=True)
def stability_records():
    from observation_inputs import arrays
    fixed_traces=arrays('sequential_full_reference_traces.npz')
    fixed_samples=arrays('sequential_sentinel_raw_samples.npz')
    records=[];summ=[];sentinels=[];raw_samples={};full_traces={}
    # Independent workflow qualification covers transient returns and changing
    # spatial shape, neither of which is checked by endpoint drift alone.
    for cid in ['R04','R16','R28']:
        target=500.
        field=np.load(c.s.F/(cid+'_field.npz'));slopes=field['true_day'];indices=[int(slopes.argmax()),int(slopes.argmin())]
        for session,kind,repeat in [(1,'steady',0),(2,'returning_cloud',0),(3,'spatial_shape',0),(4,'steady_repeat',2),(5,'steady_repeat',3)]:
            start=len(records);sentstart=len(sentinels)
            ref=json.loads((W/'camera_reference'/f'{cid}_reference_calibration.json').read_text('utf8'))
            fulltime,fulltrue,fullgray,fullproxy=fixed_traces[f'{cid}_S{session:02}']
            full_traces[f'{cid}_S{session:02}']=np.stack([fulltime,fulltrue,fullgray,fullproxy])
            for group in range(8):
                t=np.arange(151);groupstart=2150+group*150+(2100 if group>=4 else 0)
                proxy=fullproxy[groupstart:groupstart+151];rb=proxy[0];re=proxy[-1];line=rb+(re-rb)*t/150
                resid=float(np.max(abs(proxy-line)/line));endpoint=abs(re-rb)/((re+rb)/2)
                camera_pass=resid<=.035 and endpoint<=.025
                for tt,rr in zip(t,proxy):records.append(dict(case=cid,session=session,kind=kind,group=group+1,
                    time_s=groupstart+int(tt),reference_proxy_lx=float(rr),linear_reference_lx=float(line[int(tt)]),
                    relative_internal_residual_pct=100*float(abs(rr-line[int(tt)])/line[int(tt)]),qualified_differential_bound_pct=3.5,
                    internal_pass=int(camera_pass),repeat_of_session=repeat))
            baseline={}
            for checkpoint in range(3):
                for side,(site,idx) in enumerate(zip(['near_boundary','far_boundary'],indices)):
                    first=[50,2750,5450][checkpoint]+50+side*1050
                    sampletimes,readings,gray,reference=fixed_samples[f'{cid}_S{session:02}_C{checkpoint+1}_{site}']
                    normal=float(readings.mean()/reference.mean())
                    if checkpoint==0:baseline[site]=normal
                    ratio=normal/baseline[site]
                    raw_samples[f'{cid}_S{session:02}_C{checkpoint+1}_{site}']=np.stack([sampletimes,readings,gray,reference])
                    sentinels.append(dict(case=cid,session=session,checkpoint=checkpoint+1,site=site,
                        x_m=float(field['points'][idx,0]),y_m=float(field['points'][idx,1]),point_repeats=200,
                        repeat_spacing_s=5,sample_start_s=int(first),sample_end_s=int(sampletimes[-1]),low_range_resolution_lx=.1,mean_point_reading_lx=float(readings.mean()),
                        mean_reference_lx=float(reference.mean()),normalized_ratio=ratio,relative_change_pct=100*abs(ratio-1),shape_pass=int(abs(ratio-1)<=.02)))
            rc=records[start:];rs=sentinels[sentstart:];good=all(x['internal_pass'] for x in rc) and all(x['shape_pass'] for x in rs)
            summ.append(dict(case=cid,session=session,kind=kind,endpoint_rule_pass=int(all(abs(rc[g*151+150]['reference_proxy_lx']-rc[g*151]['reference_proxy_lx'])/target<=.025 for g in range(8))),
                interior_max_residual_pct=max(x['relative_internal_residual_pct'] for x in rc),
                sentinel_max_change_pct=max(x['relative_change_pct'] for x in rs),session_screen_pass=int(good),
                disposition='accept' if good else 'discard_entire_session_repeat',repeat_of_session=repeat))
    write('sequential_reference_internal_records.csv',records);write('sequential_shape_sentinels.csv',sentinels);write('sequential_session_qualification.csv',summ)
    np.savez_compressed(OUT/'sequential_sentinel_raw_samples.npz',provenance=PROV,**raw_samples)
    np.savez_compressed(OUT/'sequential_full_reference_traces.npz',provenance=PROV,**full_traces)
    assert all(x['session_screen_pass']==0 for x in summ if x['kind'] in ['returning_cloud','spatial_shape'])
    budget=[]
    for n in [8,12,24]:
        fit=88*n;protection=2760;field=456;mix=144;research=9984
        for workflow,validation in [('implementation_check_only',field+mix),('research_dense_qualification',research+field+mix)]:
            total=fit+protection+validation
            budget.append(dict(n_points=n,workflow=workflow,fit_readings=fit,protection_readings=protection,
                implementation_check_readings=field+mix,research_acceptance_readings=research if workflow=='research_dense_qualification' else 0,
                spatial_readings_total=total,spatial_hours=total*50/3600,qualification_inherited_from_24_points=0,
                sentinel_readings_per_daylight_session=1200,manual_reference_readings_per_daylight_session=2,
                sentinel_repeated_reading_spacing_s=5,sentinel_visits_per_session=6,
                camera_trace_seconds_per_nominal_session=7600,extra_sentinel_reference_hours_per_session=(1200*5+6*50+2*50)/3600,
                scope='base spatial-read/move time only; repeats, waiting, camera, meter and deployment excluded'))
    write('commissioning_labour_scope.csv',budget)
    return summ
def differential_reference_validation():
    rows=[]
    for cid in ['R04','R16','R28']:
        ref=json.loads((W/'camera_reference'/f'{cid}_reference_calibration.json').read_text('utf8'));rng=np.random.default_rng(20261480+int(cid[1:]))
        for D in [200,500,800]:
            for frac in [-.025,0,.025,.12]:
                for rep in range(1,9):
                    true=np.array([D,D*(1+frac)]);gray=c.s.p.camera_forward(true[:,None])+rng.normal(0,.7,(2,10))
                    pred=np.interp(gray,ref['gray_knots'],ref['lux_knots']).mean(1);change=(pred[1]-pred[0])/pred[0]
                    error=100*abs(change-frac)
                    rows.append(dict(case=cid,reference_lx=D,true_change_fraction=frac,repeat=rep,
                        samples_per_endpoint=10,predicted_change_fraction=float(change),absolute_error_pp=float(error),
                        declared_differential_bound_pp=3.5,qualified=int(error<=3.5)))
    write('reference_differential_validation.csv',rows)
    assert all(r['qualified'] for r in rows)
    return rows
def sentinel_precision_validation():
    from observation_inputs import arrays
    fixed_samples=arrays('sentinel_precision_raw_samples.npz')
    rows=[];raw={}
    for cid in ['R04','R16','R28']:
        ref=json.loads((c.s.R/f'{cid}_reference_calibration.json').read_text('utf8'));z=np.load(c.s.F/(cid+'_field.npz'))
        for D in [500.,800.]:
            for shape in [-.025,0.,.025,.075]:
                for rep in range(8):
                    for site,df in [('near',z['true_day'].max()),('far',z['true_day'].min())]:
                        true_ref=np.array([D,D*(1+(.025 if rep%2 else -.025))]);true_point=df*true_ref*np.array([1,1+shape])
                        sample=fixed_samples[f'{cid}_D{int(D)}_S{shape}_R{rep+1}_{site}']
                        meter,gray,cam=sample[:2],sample[2:4],sample[4:6]
                        ratio=(meter[1].mean()/cam[1].mean())/(meter[0].mean()/cam[0].mean());error=100*abs(ratio-1-shape)
                        key=f'{cid}_D{int(D)}_S{shape}_R{rep+1}_{site}';raw[key]=np.concatenate([meter,gray,cam],axis=0)
                        rows.append(dict(case=cid,target_D_lx=D,site=site,true_shape_change=shape,repeat=rep+1,
                            mean_min_signal_lx=float(true_point.min()),point_readings_per_visit=200,low_range_resolution_lx=.1,
                            normalized_ratio=float(ratio),absolute_shape_error_pp=float(error),declared_shape_error_bound_pp=3.,qualified=int(error<=3)))
    write('sentinel_precision_validation.csv',rows);np.savez_compressed(OUT/'sentinel_precision_raw_samples.npz',provenance=PROV,**raw)
    assert all(r['qualified'] for r in rows)
    return rows
def energy_evidence():
    root=W/'metering_data/metering_example'
    from metering_analysis import process
    _,models,counts=process(root);model=models['primary'];cfg=json.loads((root/'meter_settings.json').read_text('utf8'))
    terms=['quantization_fixed_bound_kWh','clock_fixed_bound_kWh','common_gain_fixed_bound_kWh','state_bias_worst_fixed_bound_kWh','residual_drift_fixed_bound_kWh']
    write('energy_fixed_bound_components.csv',[dict(component=k,value_kWh=model[k]) for k in terms]+[dict(component='sum',value_kWh=sum(model[k] for k in terms))])
    allocations=c.read(root/'assigned_days.csv');write('energy_day_allocation.csv',[{k:r[k] for k in ['day','block','position','policy']} for r in allocations])
    save('energy_statistics_definition.json',dict(n=model['n'],rank=model['rank'],df=model['df'],confidence_level=.95,
        residual_SD_kWh=model['residual_SD_kWh'],SE_kWh=model['SE_kWh'],estimate_kWh=model['estimate_kWh'],
        t_CI=[model['CI_low_kWh'],model['CI_high_kWh']],HC3_CI=[model['HC3_CI_low_kWh'],model['HC3_CI_high_kWh']],
        fixed_bound_kWh=model['fixed_envelope_kWh'],expanded_interval=[model['expanded_interval_low_kWh'],model['expanded_interval_high_kWh']],
        parameters=cfg,allocation='fixed recorded within-block sequences; not randomised; each block contains all four policies',
        initial_state='all circuits off; independent controller state each 14 h day',
        primary_assumptions='block plus strategy additive mean, independent day residuals with common variance; HC3 sensitivity reported'))
    return model
def controlled_scans():
    rows=[]
    for case in c.read(c.s.F/'room_scenarios.csv'):
        cid=case['case'];z,pool,field=c.load_case(case);cal=c.s.fit(z,pool,c.s.selections()['depth',12],'improved')
        raw=np.load(BASE/(cid+'_protection_observations.npz'));obs=dict(points=raw['points'],zones=raw['zones'],night=raw['night_readings'],day=raw['day_readings'],reference=raw['reference_readings'])
        ref=json.loads((c.s.R/(cid+'_reference_calibration.json')).read_text('utf8'));M=ref['margin_lx'];edges=c.EDGES[3]
        guard=c.protect(field,cal,obs);guard['metric']=c.metric_guard(cal,obs,edges,M)
        families={
            'prediction_no_guard':c.deficit(cal['pred'],cal['dayhat'],c.s.m.ZONES,edges,M),
            'true_same_interval':c.deficit(z['true_H'],z['true_day'],c.s.m.ZONES,edges,M),
            'task_metric_guard':c.deficit(cal['pred'],cal['dayhat'],c.s.m.ZONES,edges,M,guard)}
        lamp=c.s.m.STATES@cal['pred'].T;true=c.s.m.STATES@z['true_H'].T
        uniform=np.zeros((3,16,4));instant=np.zeros_like(uniform)
        for db,(lo,hi) in enumerate(zip(edges,np.r_[edges[1:]+2*M,1260])):
            for zone in range(4):
                ix=c.s.m.ZONES==zone;a=lamp[:,ix]+lo*cal['dayhat'][ix];b=lamp[:,ix]+hi*cal['dayhat'][ix]
                mm=cal['margin']+(cal['day_margin'] if lo>0 else 0)
                U=np.minimum((a.min(1)-mm)/np.maximum(a.mean(1)+mm,1),(b.min(1)-mm)/np.maximum(b.mean(1)+mm,1))
                uniform[db,:,zone]=np.maximum.reduce([np.zeros(16),(500-a.mean(1)+mm)/500,(300-a.min(1)+mm)/300,(.6-U)/.6])
                t=true[:,ix]+lo*z['true_day'][ix];U=t.min(1)/np.maximum(t.mean(1),1)
                instant[db,:,zone]=np.maximum.reduce([np.zeros(16),(500-t.mean(1))/500,(300-t.min(1))/300,(.6-U)/.6])
        families.update(uniform_margin=uniform,true_instant_lower=instant)
        for name,ds in families.items():
            for db in range(3):
                for req in range(1,16):
                    ix=c.s.m.STATES[req].astype(bool);feasible=(ds[db][:,ix]<=1e-12).all(1)
                    rows.append(dict(case=cid,family=name,day_bin=db,request=req,feasible_states=int(feasible.sum())))
    write('controlled_feasibility_family.csv',rows)
    write('controlled_feasibility_family_summary.csv',[dict(family=key,total_conditions=sum(r['family']==key for r in rows),feasible_conditions=sum(r['family']==key and r['feasible_states']>0 for r in rows)) for key in families])
def field_execution_order():
    from observation_inputs import records
    out=records('field_night_sequence.csv')
    assert len(out)==3*16*24
    write('field_night_sequence.csv',out)
def main():
    OUT.mkdir(exist_ok=True)
    cam=camera_qualification();ex=decision_examples();paired=paired_envelopes();scene_and_input_maps();st=stability_records();differential_reference_validation();sentinel_precision_validation();energy=energy_evidence();controlled_scans();field_execution_order();rep=replay_qualification()
    save('reference_validation_summary.json',dict(provenance=PROV,camera_qualified_records=sum(x['qualified'] for x in cam),
        camera_range_exclusions=sum(x['excluded'] for x in cam),camera_tail_qualification_pass=True,
        replay_configurations=len(rep),changed_frames=sum(x['changed_frames'] for x in rep),
        interval_violations=sum(x['qualified_seconds']-x['contained_seconds'] for x in rep),
        R30=next(x for x in paired if x['case']=='R30'),workflow_sessions=len(st),workflow_accepted=sum(x['session_screen_pass'] for x in st),
        energy_fixed_bound_kWh=energy['fixed_envelope_kWh'],scope='study main and S8 regenerated under qualified reference; additional uncertainty evidence'))
    print('study complete',flush=True)
if __name__=='__main__':main()
