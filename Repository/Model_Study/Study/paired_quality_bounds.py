"""Shared-evaluation-grid paired outer bounds with explicit invalid-reference scope."""
import json,numpy as np
import analysis_core as c

def point_bounds(cal,g,z,margin):
    lows=np.r_[c.EDGES[3],0.];highs=np.r_[c.EDGES[3][1:]+2*margin,1260.,1260.]
    night=c.s.m.STATES@cal['final_pred'].T
    true_night=c.s.m.STATES@z['final_H'].T
    possible=np.zeros((4,16,4),bool);certain=np.zeros_like(possible);details=[];proof=[]
    for db,(lo,hi) in enumerate(zip(lows,highs)):
        for zone in range(4):
            ix=z['final_zones']==zone;df=cal['final_dayhat'][ix]
            extra=.125*(abs(df)+max(g['dplus'][zone],g['dminus'][zone]))
            dp=g['dplus'][zone]+extra;dm=g['dminus'][zone]+extra
            L=np.maximum(0,night[:,ix]+lo*df-g['nplus'][:,zone,None]-lo*dp)
            V=np.maximum(0,night[:,ix]+hi*df+g['nminus'][:,zone,None]+hi*dm)
            lm,ln=L.mean(1),L.min(1);vm,vn=V.mean(1),V.min(1)
            ulo=ln/np.maximum(vm,1);uhi=np.minimum(1,vn/np.maximum(lm,1))
            possible[db,:,zone]=(lm<500)|(ln<300)|(ulo<.6)
            certain[db,:,zone]=(vm<500)|(vn<300)|(uhi<.6)
            actual_low=true_night[:,ix]+lo*z['final_day'][ix]
            actual_hi=true_night[:,ix]+hi*z['final_day'][ix]
            extended_low=true_night[:,ix]+lo*z['final_day'][ix]*(1-.07)*(1-.05)
            extended_hi=true_night[:,ix]+hi*z['final_day'][ix]*(1+.07)*(1+.05)
            proof.append(dict(day_bin=db,zone=zone+1,daylight_lower_lx=float(lo),daylight_upper_lx=float(hi),
                evaluation_points=52,endpoint_values=2*16*52,
                lower_exceed_values=int((L>actual_low+1e-8).sum()),upper_exceed_values=int((V<actual_hi-1e-8).sum()),
                minimum_lower_slack_lx=float((actual_low-L).min()),minimum_upper_slack_lx=float((V-actual_hi).min()),
                extension_endpoint_values=2*16*52,extension_multiplier_low=.8835,extension_multiplier_high=1.1235,
                extended_lower_exceed_values=int((L>extended_low+1e-8).sum()),extended_upper_exceed_values=int((V<extended_hi-1e-8).sum())))
            for st in range(16):
                details.append(dict(day_bin=db,reference_role='invalid_full_declared_domain' if db==3 else 'qualified_camera_bin',zone=zone+1,state=st,
                    daylight_lower_lx=float(lo),daylight_upper_lx=float(hi),evaluation_points=52,
                    mean_lower_lx=float(lm[st]),mean_upper_lx=float(vm[st]),minimum_lower_lx=float(ln[st]),minimum_upper_lx=float(vn[st]),
                    uniformity_lower=float(ulo[st]),uniformity_upper=float(uhi[st]),certain_shortfall=int(certain[db,st,zone]),possible_shortfall=int(possible[db,st,zone])))
    return possible,certain,details,proof

def main():
    from quality_analysis import automatic_reading,qualified_table
    inputs={d:c.s.m.construct(20261339+d) for d in range(1,9)};design=c.s.selections()
    nominal=c.read(c.OUT/'current_contrasts_daily.csv');rows=[];aggregate=[];proofs=[];boxes=[];example=[]
    for case in c.read(c.s.F/'room_scenarios.csv'):
        cid=case['case'];z,pool,field=c.load_case(case)
        raw=np.load(c.OUT/(cid+'_protection_observations.npz'))
        obs=dict(points=raw['points'],zones=raw['zones'],night=raw['night_readings'],day=raw['day_readings'],reference=raw['reference_readings'])
        ref=json.loads((c.s.R/(cid+'_reference_calibration.json')).read_text('utf8'));tr=np.load(c.OUT/(cid+'_reference_traces.npz'))
        pp={};cc={}
        for n in [12,24]:
            cal=c.s.fit(z,pool,design['depth',n],'improved');g=c.protect(field,cal,obs)
            pp[n],cc[n],detail,proof=point_bounds(cal,g,z,ref['margin_lx'])
            boxes.extend(dict(case=cid,n_points=n,**r) for r in detail)
            proofs.extend(dict(case=cid,n_points=n,**r) for r in proof)
        for day,inp in inputs.items():
            auto=automatic_reading(inp);den=5*int(auto.sum());ids={n:tr[f'D{day:02}_{n}'] for n in [12,24]}
            tables={};absolute={};invalid_counts={}
            table,_,_=qualified_table(inp,ref,c.EDGES[3])
            for n in [12,24]:
                db=table[np.arange(c.s.m.SECONDS),ids[n][::5]]
                # An invalid reference does not identify bin zero. Its available
                # daylight information is exactly the externally declared domain.
                tables[n]=np.where(db<0,3,db)
                invalid_counts[n]=sum(int((auto&(db<0)[:,None]).sum()) for _ in range(5))
                lower=upper=0
                for phase in range(5):
                    st=ids[n][phase::5]
                    lower+=int((cc[n][tables[n],st]&auto).sum());upper+=int((pp[n][tables[n],st]&auto).sum())
                absolute[n]=(100*lower/max(den,1),100*upper/max(den,1))
            low=up=changed=0;shared=0
            for phase in range(5):
                a=ids[12][phase::5];b=ids[24][phase::5];different=(a!=b)[:,None]&auto
                lc=(cc[24][tables[24],b].astype(int)-pp[12][tables[12],a].astype(int))*different
                uc=(pp[24][tables[24],b].astype(int)-cc[12][tables[12],a].astype(int))*different
                low+=int(lc.sum());up+=int(uc.sum());changed+=int(different.sum());shared+=int((((a==b)[:,None])&auto).sum())
                if cid=='R10' and day==1 and phase==0:
                    candidates=np.flatnonzero(different.any(1))
                    for sec in candidates[:8]:
                        for zone in np.flatnonzero(auto[sec]):
                            example.append(dict(case=cid,day=day,time_s=int(sec),phase=phase,zone=int(zone+1),state12=int(a[sec]),state24=int(b[sec]),bin12=int(tables[12][sec]),bin24=int(tables[24][sec]),certain12=int(cc[12][tables[12][sec],a[sec],zone]),possible12=int(pp[12][tables[12][sec],a[sec],zone]),certain24=int(cc[24][tables[24][sec],b[sec],zone]),possible24=int(pp[24][tables[24][sec],b[sec],zone]),lower_difference=int(lc[sec,zone]),upper_difference=int(uc[sec,zone])))
            nom=float(next(r['Jauto_difference_pp'] for r in nominal if (r['case'],r['layout'],r['n_points'],r['policy'],r['day'])==(cid,'depth','24','PU',str(day))))
            row=dict(case=cid,day=day,J12_lower_pct=absolute[12][0],J12_upper_pct=absolute[12][1],J24_lower_pct=absolute[24][0],J24_upper_pct=absolute[24][1],nominal_delta_J_pp=nom,paired_lower_delta_J_pp=100*low/max(den,1),paired_upper_delta_J_pp=100*up/max(den,1),changed_reading_exposure_pct=100*changed/max(den,1),auto_reading_zone_frames=den,shared_state_reading_frames=shared,same_state_lower_contribution=0,same_state_upper_contribution=0,reference_invalid_exposure_frames12=invalid_counts[12],reference_invalid_exposure_frames24=invalid_counts[24],evaluation_positions=208,points_per_zone=52,sequential_reference_extra_fraction=.07,shape_extra_fraction=.05,combined_extra_fraction=.125,shape_bound_role='independent full-task-plane design condition; not inferred from sentinel passing',interpretation='conditional shared-grid outer interval; invalid reference uses [0,1260]; not confidence interval')
            row['nominal_contained']=int(row['paired_lower_delta_J_pp']-1e-9<=nom<=row['paired_upper_delta_J_pp']+1e-9)
            assert row['nominal_contained'],(cid,day,row)
            rows.append(row)
        selected=[r for r in rows if r['case']==cid]
        aggregate.append(dict(case=cid,input_days=8,evaluation_positions=208,nominal_delta_J_pp=float(np.mean([r['nominal_delta_J_pp'] for r in selected])),paired_lower_delta_J_pp=float(np.mean([r['paired_lower_delta_J_pp'] for r in selected])),paired_upper_delta_J_pp=float(np.mean([r['paired_upper_delta_J_pp'] for r in selected])),daily_nominal_contained=sum(r['nominal_contained'] for r in selected),same_state_contribution=0,reference_invalid_exposure_frames12=sum(r['reference_invalid_exposure_frames12'] for r in selected),reference_invalid_exposure_frames24=sum(r['reference_invalid_exposure_frames24'] for r in selected)))
        print('study paired grid',cid,flush=True)
    c.write('paired_quality_envelopes.csv',rows);c.write('paired_grid_validation.csv',aggregate)
    c.write('paired_point_endpoint_validation.csv',proofs);c.write('paired_bound_metric_details.csv',boxes);c.write('paired_nonzero_example.csv',example)
    current=[]
    for r in c.read(c.OUT/'commissioning_decision_screen.csv'):
        q=next(x for x in aggregate if x['case']==r['case']);upper=q['paired_upper_delta_J_pp']
        current.append(dict(**r,paired_lower_delta_J_pp=q['paired_lower_delta_J_pp'],paired_upper_delta_J_pp=upper,quality_improvement_identified=int(upper<=-.2),quality_noninferiority_identified=int(upper<=.2),interpretation='conditional on independent full-plane shape bound; nominal design candidate only' if 'gain_candidate' in r['decision'] else 'descriptive design result'))
    c.write('commissioning_decision_uncertainty.csv',current)
    assert not any(r['lower_exceed_values'] or r['upper_exceed_values'] for r in proofs)
    assert not any(r['extended_lower_exceed_values'] or r['extended_upper_exceed_values'] for r in proofs)
    return current

if __name__=='__main__':main()
