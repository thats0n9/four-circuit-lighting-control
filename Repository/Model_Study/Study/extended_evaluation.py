"""Independent optical-form checks, session sensitivity and field sampling records."""
import argparse,json,math,time,copy
from pathlib import Path
import numpy as np
import analysis_core as c
from analysis_engine import fast_engine_trace
s=c.s
CASES=['R04','R16','R28']

def optical_variant(field,kind):
    xyz,norm,area,B0,src,sn,sw0,ref0=field['solver']
    sw=sw0.copy();L,W,C=field['geometry']
    if kind=='window_nonuniform':
        # An independently specified transverse attenuation, absent from basis.
        sw[128:,4]*=np.where(src[128:,1]<.45*W,.25,1.)
    q=1.6 if kind=='angular_1p6' else 1.
    def kernel(target,tn):
        k=s.v.coupling(target,tn,src,sn,np.ones(len(src)))
        if q!=1:
            rr=src[None]-target[:,None];rr2=np.sum(rr*rr,2)
            co=np.maximum(-np.einsum('ijk,jk->ij',rr,sn)/np.sqrt(np.maximum(rr2,1e-12)),0)
            k[:,:128]*=(q+1)/2*co[:,:128]**(q-1)
        return k
    F0=s.v.coupling(xyz,norm,xyz,norm,area);S=area[:,None]*F0;bal=np.ones(len(area))
    for _ in range(400):
        bal*=np.sqrt(area/np.maximum(bal*(S@bal),1e-30))
        if np.max(abs(bal*(S@bal)/area-1))<1e-10:break
    FF=bal[:,None]*S*bal[None]/area[:,None]
    E0base=s.v.coupling(xyz,norm,src,sn,np.ones(len(src)))@sw0
    denom=E0base+FF@B0
    rho=np.mean(np.divide(B0,denom,out=np.zeros_like(B0),where=denom>1e-12),axis=1)
    E0=kernel(xyz,norm)@sw;B=rho[:,None]*E0
    for _ in range(300):
        nxt=rho[:,None]*(E0+FF@B)
        if np.max(abs(nxt-B))<1e-8:B=nxt;break
        B=nxt
    ref=np.array([[.5*L/11,W/2,.75]])
    rv=(kernel(ref,np.array([[0.,0.,1.]]))@sw+s.v.coupling(ref,np.array([[0.,0.,1.]]),xyz,norm,area)@B)[0,4]
    def at(points):
        p=np.column_stack([points,np.full(len(points),.75)]);up=np.tile([0.,0.,1.],(len(points),1))
        e=kernel(p,up)@sw+s.v.coupling(p,up,xyz,norm,area)@B
        return e[:,:4],e[:,4]/rv
    return at,dict(kind=kind,angular_exponent=q,normalized_lamp_flux_multiplier=(q+1)/2,
                   window_transverse_attenuation=.25 if kind=='window_nonuniform' else 1.,
                   radiosity_equation_residual=float(abs(B-rho[:,None]*(E0+FF@B)).max()))

def observed_pool(z,base,seed,gain=1.):
    pool={k:np.array(v,copy=True) for k,v in base.items()};rng=np.random.default_rng(seed)
    nt=s.m.STATES@z['true_H'].T
    pool['night']=gain*np.maximum(0,nt[None]+rng.normal(0,.008*nt[None]+.7,size=(4,16,81)))
    levels=np.array([100.,250.,500.,800.]);day=levels[:,None]*z['true_day']
    pool['day']=gain*np.maximum(.1,day[None]+rng.normal(0,.008*day[None]+.7,size=(6,4,81)))
    pool['reference']=gain*np.maximum(.1,levels[None]+rng.normal(0,.008*levels[None]+.7,size=(6,4)))
    return pool

def protection_obs(case,H,day,seed,gain=1.):
    points,zones=c.protection_points(case);rng=np.random.default_rng(seed)
    nt=s.m.STATES@H.T;levels=np.array([100.,250.,500.,800.]);dt=levels[:,None]*day
    return dict(points=points,zones=zones,
        night=gain*np.maximum(0,nt[None]+rng.normal(0,.008*nt[None]+.7,size=(2,16,69))),
        day=gain*np.maximum(0,dt[None]+rng.normal(0,.008*dt[None]+.7,size=(2,4,69))),
        reference=gain*np.maximum(.1,levels[None]+rng.normal(0,.008*levels[None]+.7,size=(2,4))),true_H=H,true_day=day)

def off_model():
    rows=[];coverage=[];params=[]
    for cid in CASES:
        case=next(r for r in c.read(s.F/'room_scenarios.csv') if r['case']==cid)
        z0,base,_=c.load_case(case)
        field=s.v.radiosity_field(float(case['length_m']),float(case['width_m']),float(case['ceiling_m']),
                   'low' if case['wall_reflectance']=='0.35' else 'high',case['window'])
        pts,pzones=c.protection_points(case)
        ref=json.loads((s.R/(cid+'_reference_calibration.json')).read_text('utf8'))
        for kind in ['angular_1p6','window_nonuniform']:
            at,diag=optical_variant(field,kind);params.append(dict(case=cid,**diag))
            z=copy.deepcopy(z0);h,d=at(z['points']);fh,fd=at(z['final_points'])
            z['true_H']=h*float(case['flux_scale']);z['true_day']=d
            z['final_H']=fh*float(case['flux_scale']);z['final_day']=fd
            pool=observed_pool(z,base,int(case['calibration_seed'])+1310000)
            hp,dp=at(pts);obs=protection_obs(case,hp*float(case['flux_scale']),dp,int(case['calibration_seed'])+1320000)
            # Fitting basis stays NOMINAL; new truth never enters the basis.
            np.savez_compressed(c.OUT/(cid+'_'+kind+'_optical_inputs.npz'),provenance=c.PROV,
                  true_H=z['true_H'],true_day=d,final_H=z['final_H'],final_day=fd,
                  pool_night=pool['night'],pool_day=pool['day'],pool_reference=pool['reference'],
                  protection_night=obs['night'],protection_day=obs['day'],protection_reference=obs['reference'])
            for layout in s.LAYOUTS:
                for n in [8,12,24]:
                    for method in (['original','improved'] if layout=='depth' and n==12 else ['improved']):
                        cal=s.fit(z,pool,s.selections()[layout,n],method)
                        predicted=s.m.STATES@cal['final_pred'].T+np.array([0,200,500])[:,None,None]*cal['final_dayhat'][None,None]
                        actual=s.m.STATES@z['final_H'].T+np.array([0,200,500])[:,None,None]*z['final_day'][None,None]
                        gg={'metric':c.metric_guard(cal,obs,c.EDGES[3],ref['margin_lx']),
                            'nplus':np.zeros((16,4)),'nminus':np.zeros((16,4)), 'dplus':np.zeros(4),'dminus':np.zeros(4)}
                        mv=c.metric_coverage(z,cal,gg,c.EDGES[3],ref['margin_lx'],layout,n,cid)
                        for rr in mv:rr['optical_condition']=kind;rr['method']=method
                        coverage+=mv
                        rows.append(dict(provenance=c.PROV,case=cid,optical_condition=kind,layout=layout,n_points=n,method=method,
                          combined_truth_RMSE_lx=float(np.sqrt(np.mean((predicted-actual)**2))),
                          daylight500_RMSE_lx=float(np.sqrt(np.mean((500*(cal['final_dayhat']-fd))**2))),
                          night_RMSE_lx=float(np.sqrt(np.mean((s.m.STATES@(cal['final_pred']-z['final_H']).T)**2))),
                          minimum_metric_coverage_pct=min(r['one_sided_coverage_pct'] for r in mv),
                          false_accept_zone_state_levels=sum(r['false_accept_zone_state_levels'] for r in mv if r['metric']=='mean')))
        print('independent optical forms',cid,flush=True)
    c.write('off_model_spatial_metrics.csv',rows);c.write('off_model_metric_coverage.csv',coverage);c.write('off_model_parameters.csv',params)

def sensitivity():
    rows=[];cov=[];trace=[]
    for cid in CASES:
        case=next(r for r in c.read(s.F/'room_scenarios.csv') if r['case']==cid)
        z,base,_=c.load_case(case)
        ref0=json.loads((s.R/(cid+'_reference_calibration.json')).read_text('utf8'))
        field=s.v.radiosity_field(float(case['length_m']),float(case['width_m']),float(case['ceiling_m']),
                    'low' if case['wall_reflectance']=='0.35' else 'high',case['window'])
        orig_obs=np.load(c.OUT/(cid+'_protection_observations.npz'))
        obs0=dict(points=orig_obs['points'],zones=orig_obs['zones'],night=orig_obs['night_readings'],
                  day=orig_obs['day_readings'],reference=orig_obs['reference_readings'],true_H=orig_obs['true_H'],true_day=orig_obs['true_day'])
        models={}
        for label,gain,rep in [('nominal',1.,0),('common_gain097',.97,0),('common_gain103',1.03,0)]+[(f'new_session{j}',1.,j) for j in range(1,5)]:
            pool=base if rep==0 else observed_pool(z,base,int(case['calibration_seed'])+1330000+rep*1000)
            obs=copy.deepcopy(obs0) if rep==0 else protection_obs(case,obs0['true_H'],obs0['true_day'],int(case['calibration_seed'])+1340000+rep*1000)
            if gain!=1:
                pool=copy.deepcopy(pool)
                for k in ['night','day','reference']:pool[k]=pool[k]*gain;obs[k]=obs[k]*gain
            ref=copy.deepcopy(ref0)
            for k in ['lux_knots','lamp_camera_equivalent_lx']:ref[k]=(np.array(ref[k])*gain).tolist()
            ref['margin_lx']*=gain
            for n in [8,12,24]:
                cal=s.fit(z,pool,s.selections()['depth',n],'improved')
                g=c.protect(field,cal,obs);g['metric']=c.metric_guard(cal,obs,c.EDGES[3],ref['margin_lx'])
                lk,dz,ff,uf=c.lookups(c.deficit(cal['pred'],cal['dayhat'],s.m.ZONES,c.EDGES[3],ref['margin_lx'],g))
                models[label,n]=(ref,lk,dz)
                mv=c.metric_coverage(z,cal,g,c.EDGES[3],ref['margin_lx'],'depth',n,cid)
                for r in mv:r['session_condition']=label
                cov+=mv
        for day,seed in enumerate(range(20261360,20261364),1):
            inp=s.m.construct(seed);qt=c.quality_tables(inp,z);manual=np.array(inp['manuals'],np.int64).reshape(-1,4)
            for (label,n),(ref,lk,dz) in models.items():
                table=s.r.bins_table(inp,ref,c.EDGES[3])
                for pol,look in [('PU',lk),('DZU',dz)]:
                    ids,vi,bi=s.fast_engine(inp['det'],inp['valid'],table,look,manual)
                    rows.append(dict(provenance=c.PROV,case=cid,session_condition=label,n_points=n,policy=pol,
                           day=day,input_seed=seed,**c.evaluate(ids,inp,z,qt,vi)))
        print('whole-pipeline sensitivity',cid,flush=True)
    c.write('session_sensitivity_daily.csv',rows);c.write('session_sensitivity_coverage.csv',cov)

def field_sampling():
    from observation_inputs import records
    rows=records('field_sequential_daylight_records.csv')
    night=records('field_terminal_night_records.csv')
    mixed=records('field_mixed_additivity_records.csv')
    stability=records('field_stability_sessions.csv');budget=[]
    for row in rows:
        start,end,at=[float(row[key]) for key in ['reference_start_time_s','reference_end_time_s','reading_time_s']]
        rb,re=[float(row[key]) for key in ['reference_start_lx','reference_end_lx']]
        interp=rb+(re-rb)*(at-start)/(end-start)
        drift=abs(re-rb)/max((re+rb)/2,1)
        assert abs(interp-float(row['interpolated_reference_lx']))<1e-10
        assert abs(float(row['raw_reading_lx'])*float(row['target_reference_lx'])/interp-float(row['normalized_reading_lx']))<1e-10
        assert int(drift<=.025)==int(row['eligible'])
    for row in mixed:
        reading=float(row['observed_combined_lx']);combined=float(row['component_sum_lx'])
        assert abs(reading-combined-float(row['residual_lx']))<1e-10
        assert int(abs(reading-combined)<=.03*reading+2)==int(row['eligible'])
    for n in [8,12,24]:
        fit=88*n;protection=69*(2*16+2*4);terminal=24*(16+3);additive=12*4*3
        reference=75+64+512
        budget.append(dict(point_count=n,fit_point_readings=fit,shared_protection_point_readings=protection,
            shared_terminal_point_readings=terminal,mixed_additivity_readings=additive,
            camera_reference_records=reference,spatial_readings_total=fit+protection+terminal+additive,
            spatial_reading_move_hours=(fit+protection+terminal+additive)*50/3600,
            marginal_12_to24_hours=88*12*50/3600,shared_checks_hours=(protection+terminal+additive)*50/3600,
            scope='50s per spatial reading; settling, reference brackets, camera, power and weather wait separate'))
    c.write('field_sequential_daylight_records.csv',rows);c.write('field_stability_sessions.csv',stability)
    c.write('field_terminal_night_records.csv',night)
    c.write('field_mixed_additivity_records.csv',mixed);c.write('commissioning_labour_budget.csv',budget)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['off-model','sensitivity','field','all'],default='all');a=ap.parse_args()
    if a.part in ['off-model','all']:off_model()
    if a.part in ['sensitivity','all']:sensitivity()
    if a.part in ['field','all']:field_sampling()

if __name__=='__main__':main()
