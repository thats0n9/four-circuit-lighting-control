"""Numerical model and evaluation routines."""
from pathlib import Path
import json,numpy as np
import analysis_core as c
from analysis_engine import fast_engine_trace
D=c.OUT;W=c.W

def inputs():
    return {j:dict(np.load(D/f'input_{20261339+j}.npz')) for j in range(1,9)}

def linear_pass_interval(intercept,slope):
    positive=slope>1e-14;negative=slope<-1e-14;zero=~(positive|negative)
    if np.any(intercept[zero]<0):return np.inf,-np.inf
    lo=max(0.,float(np.max(-intercept[positive]/slope[positive])) if positive.any() else 0.)
    hi=float(np.min(-intercept[negative]/slope[negative])) if negative.any() else np.inf
    return lo,hi

def quality_from_state_field(night,day,z,inp):
    # Every task inequality is affine in scalar D. Its simultaneous pass set is
    # an interval, evaluated exactly at every held second, not a coarse D grid.
    joint=np.zeros((16,c.s.m.SECONDS,4),bool)
    for zone in range(4):
        ix=z['final_zones']==zone
        for state in range(16):
            n=night[state,ix];d=day[ix];nm=n.mean();dm=d.mean()
            a=np.r_[nm-500,n-300,n-.6*nm,n-.6]
            b=np.r_[dm,d,d-.6*dm,d]
            lo,hi=linear_pass_interval(a,b)
            joint[state,:,zone]=(inp['window']<lo)|(inp['window']>hi)
            # Independent direct-field checks at fixed levels ensure that the
            # interval transformation preserves mean, minimum and U conventions.
            levels=np.array([0,40,200,500,800,1260],float)
            e=n[None]+levels[:,None]*d[None];mean=e.mean(1);minimum=e.min(1)
            direct=(mean<500)|(minimum<300)|(minimum/np.maximum(mean,1)<.6)
            assert np.array_equal(direct,(levels<lo)|(levels>hi))
    active=inp['work'].astype(bool);manual=np.zeros_like(active)
    for t,zone,value,duration in inp['manuals']:
        if value==0:manual[t//5:min(len(active),(t+duration)//5),zone]=True
    return joint,active,active&~manual

def evaluate_joint(ids,qt):
    joint,active,auto=qt;sec=np.arange(len(active));all_bad=auto_bad=0
    for phase in range(5):
        actual=joint[ids[phase::5],sec]
        all_bad+=int((actual&active).sum());auto_bad+=int((actual&auto).sum())
    return dict(J_pct=100*all_bad/max(5*int(active.sum()),1),Jauto_pct=100*auto_bad/max(5*int(auto.sum()),1))

def terminal_reconstruction_and_reading():
    ins=inputs();daily=[];reconstruction=[];source_daily=c.read(D/'current_control_daily.csv')
    for case in c.read(c.s.F/'room_scenarios.csv'):
        cid=case['case'];raw=np.load(D/(cid+'_final_validation.npz'))
        y=raw['validation_readings'];levels=np.array([0.,200.,500.])
        # Independent noisy reconstruction: retain each state's measured night
        # contribution and estimate a common slope from its day-night differences.
        night=y[0];slope=np.maximum(0,np.sum(levels[1:,None,None]*(y[1:]-night[None]),axis=(0,1))/(16*np.sum(levels[1:]**2)))
        true_night=c.s.m.STATES@raw['true_H'].T;truth_day=raw['true_day']
        np.savez_compressed(D/(cid+'_terminal_reconstruction.npz'),points=raw['points'],zones=raw['zones'],
            observation_levels_lx=levels,observed_night_by_state=night,reconstructed_day_slope=slope,
            benchmark_night_by_state=true_night,benchmark_day_slope=truth_day,provenance=c.PROV)
        reconstruction.append(dict(case=cid,positions=208,points_per_zone=52,original_observations=9984,
            night_RMSE_lx=float(np.sqrt(np.mean((night-true_night)**2))),day_slope_RMSE=float(np.sqrt(np.mean((slope-truth_day)**2))),
            reconstruction500_RMSE_lx=float(np.sqrt(np.mean((night+500*slope-true_night-500*truth_day)**2))),
            reconstruction1260_RMSE_lx=float(np.sqrt(np.mean((night+1260*slope-true_night-1260*truth_day)**2))),
            nominal_endpoint_role='independent-position noisefree benchmark; noisy reconstruction is sensitivity only'))
        z=dict(final_zones=raw['zones']);tr=np.load(D/(cid+'_reference_traces.npz'))
        for day,inp in ins.items():
            qt=quality_from_state_field(night,slope,z,inp)
            for n in [12,24]:
                baseline=next(r for r in source_daily if (r['case'],r['layout'],r['n_points'],r['policy'],r['day'])==(cid,'depth',str(n),'PU',str(day)))
                ids=tr[f'D{day:02}_{n}'];rec=evaluate_joint(ids,qt)
                row=dict(case=cid,day=day,n_points=n,policy='PU',
                    **{k:float(baseline[k]) for k in ['reading_zone_seconds','auto_reading_zone_seconds','manual_reading_zone_seconds','J_pct','Jauto_pct','Jauto_contribution_pct','Jmanual_contribution_pct']},
                    reconstructed_Jauto_pct=rec['Jauto_pct'],reconstructed_J_pct=rec['J_pct'],
                    reconstructed_minus_benchmark_Jauto_pp=rec['Jauto_pct']-float(baseline['Jauto_pct']))
                assert abs(row['Jauto_contribution_pct']+row['Jmanual_contribution_pct']-row['J_pct'])<1e-10
                assert abs(row['Jauto_pct']*row['auto_reading_zone_seconds']/row['reading_zone_seconds']-row['Jauto_contribution_pct'])<1e-10
                daily.append(row)
        print('study reconstruction',cid,flush=True)
    summary=[]
    for cid in [f'R{i:02}' for i in range(1,37)]+['ALL36']:
        for n in [12,24]:
            rr=[r for r in daily if r['n_points']==n and (cid=='ALL36' or r['case']==cid)]
            summary.append(dict(case=cid,n_points=n,case_days=len(rr),aggregation='equal day then equal case; contributions use all-reading denominator',
                **{k:float(np.mean([r[k] for r in rr])) for k in ['reading_zone_seconds','auto_reading_zone_seconds','manual_reading_zone_seconds','J_pct','Jauto_pct','Jauto_contribution_pct','Jmanual_contribution_pct','reconstructed_Jauto_pct','reconstructed_J_pct','reconstructed_minus_benchmark_Jauto_pp']}))
    contrasts=[]
    for cid in [f'R{i:02}' for i in range(1,37)]:
        a=next(r for r in summary if (r['case'],r['n_points'])==(cid,12));b=next(r for r in summary if (r['case'],r['n_points'])==(cid,24))
        contrasts.append(dict(case=cid,nominal_delta_Jauto_pp=b['Jauto_pct']-a['Jauto_pct'],reconstruction_delta_Jauto_pp=b['reconstructed_Jauto_pct']-a['reconstructed_Jauto_pct'],
            reconstruction_delta_Jall_pp=b['reconstructed_J_pct']-a['reconstructed_J_pct'],nominal_delta_Jall_pp=b['J_pct']-a['J_pct']))
    c.write('terminal_reconstruction_diagnostics.csv',reconstruction);c.write('all_reading_quality_daily.csv',daily)
    c.write('all_reading_quality_summary.csv',summary);c.write('terminal_reconstruction_contrasts.csv',contrasts)
    c.save('terminal_evaluation_roles.json',dict(provenance=c.PROV,positions=208,points_per_zone=52,raw_levels_lx=[0,200,500],raw_readings=9984,
        prediction_error='prediction minus independent noisy observation; no reconstructed field substituted',
        qualification='frozen prediction and guards compared with independent-position noisefree benchmark field',
        nominal_quality='same independent-position benchmark at held 1Hz daylight; 5Hz actual output state',
        paired_bounds='both predictions evaluated at same 208 positions; 52 equal-weight points per task zone',
        reconstruction='N_s(p)=Y_0,s,p; d_rec(p)=max(0,sum_D,s D*(Y_D,s,p-Y_0,s,p)/(16*sum_D D^2)); levels 200 and 500',
        reconstruction_use='secondary sensitivity at same output traces; extrapolation to 1260 conditional on affine response',
        measured_levels_distinct_from_model_sweep=True,model_evaluation_domain_lx=[0,1260]))
    return summary

def qualification_examples():
    scene=c.read(D/'current_scene_summary.csv')
    failure=next(r for r in scene if r['layout']=='depth' and r['n_points']=='24' and r['policy']=='PU' and r['protection_qualified']=='0' and int(r['metric_false_accept_zone_state_levels'])>0)
    selected=[('R16',12),(failure['case'],24)];all_details=[];summary=[]
    for cid,n in selected:
        case=next(r for r in c.read(c.s.F/'room_scenarios.csv') if r['case']==cid)
        z,pool,field=c.load_case(case);raw=np.load(D/(cid+'_protection_observations.npz'))
        obs=dict(points=raw['points'],zones=raw['zones'],night=raw['night_readings'],day=raw['day_readings'],reference=raw['reference_readings'])
        cal=c.s.fit(z,pool,c.s.selections()['depth',n],'improved');ref=json.loads((c.s.R/(cid+'_reference_calibration.json')).read_text('utf8'))
        metric=c.metric_guard(cal,obs,c.EDGES[3],ref['margin_lx']);pred_lamp=c.s.m.STATES@cal['pred'].T;truth_lamp=c.s.m.STATES@z['final_H'].T
        rr=[]
        for db,(lo,hi) in enumerate(zip(c.EDGES[3],np.r_[c.EDGES[3][1:]+2*ref['margin_lx'],1260])):
            levels=np.unique(np.r_[lo,np.arange(np.ceil(lo/25)*25,hi,25),hi])
            for zone in range(4):
                ix=c.s.m.ZONES==zone;tx=z['final_zones']==zone
                for level in levels:
                    p=pred_lamp[:,ix]+level*cal['dayhat'][ix];a=truth_lamp[:,tx]+level*z['final_day'][tx]
                    pm=p.mean(1);pn=p.min(1);pu=pn/np.maximum(pm,1);am=a.mean(1);an=a.min(1);au=an/np.maximum(am,1)
                    lower=np.stack([pm,pn,pu],axis=1)-metric[db,:,zone]
                    accepted=(lower>=np.array([500,300,.6])).all(1);real=(np.stack([am,an,au],axis=1)>=np.array([500,300,.6])).all(1)
                    for st in range(16):
                        rr.append(dict(case=cid,n_points=n,day_bin=db,daylight_lx=float(level),zone=zone+1,state=st,accepted=int(accepted[st]),actual_task_pass=int(real[st]),false_accept=int(accepted[st] and not real[st]),
                            planning_mean_lx=float(pm[st]),planning_min_lx=float(pn[st]),planning_uniformity=float(pu[st]),protected_mean_lx=float(lower[st,0]),protected_min_lx=float(lower[st,1]),protected_uniformity=float(lower[st,2]),
                            terminal_mean_lx=float(am[st]),terminal_min_lx=float(an[st]),terminal_uniformity=float(au[st])))
        row=next(r for r in scene if (r['case'],r['layout'],r['n_points'],r['policy'])==(cid,'depth',str(n),'PU'))
        accepted=sum(r['accepted'] for r in rr);false=sum(r['false_accept'] for r in rr)
        assert accepted==int(row['metric_accepted_zone_state_levels']) and false==int(row['metric_false_accept_zone_state_levels'])
        candidates=[r for r in rr if r['false_accept']] if false else [r for r in rr if r['accepted'] and r['actual_task_pass']]
        witness=min(candidates,key=lambda r:abs(r['terminal_uniformity']-.6))
        summary.append(dict(case=cid,n_points=n,point_component_min_coverage_pct=float(row['protection_min_zone_component_coverage_pct']),accepted_candidates=accepted,false_accept_candidates=false,research_qualified=int(row['protection_qualified']),**{'witness_'+k:v for k,v in witness.items() if k not in ['case','n_points']}))
        all_details.extend(rr)
    c.write('terminal_qualification_examples.csv',summary);c.write('terminal_qualification_example_details.csv',all_details)
    return summary

def sentinel_counterexamples():
    rows=[];arrays={};t=np.arange(7601,dtype=float)
    for cid in ['R04','R16','R28']:
        case=next(r for r in c.read(c.s.F/'room_scenarios.csv') if r['case']==cid)
        z,pool,field=c.load_case(case);near=int(np.argmax(z['true_day']));far=int(np.argmin(z['true_day']))
        sites=z['points'][[near,far]];L=float(case['length_m']);ww=float(case['width_m']);reference=np.array([.5*L/11,ww/2])
        def bump_at(points):
            ans=np.ones(len(points))
            for site in [*sites,reference]:ans*=np.sum(((points-site)/np.array([L,ww]))**2,axis=1)
            return ans
        bump=bump_at(z['final_points']);scale=float(bump.max());bump/=scale
        assert np.allclose(bump_at(np.vstack([sites,reference]))/scale,0)
        # The first two fields belong to the declared bounded examples. Hidden
        # bumps are deliberate counterexamples outside the 5% full-plane domain.
        for name,amplitude,kind in [('bounded_uniform',.015,'uniform'),('detected_far_shape',.075,'uniform'),('hidden_interior_shape',.15,'static_bump'),('returning_hidden_shape',.15,'returning_bump')]:
            shape=np.ones(208) if kind=='uniform' else bump
            temporal=np.ones_like(t)
            if kind=='returning_bump':
                temporal=np.zeros_like(t);mask=(t>=1225)&(t<=1375);temporal[mask]=np.sin(np.pi*(t[mask]-1225)/150)**2
            sentinel_shape=np.ones(2) if kind=='uniform' else np.zeros(2)
            # Reference plane stays fixed; near/far task readings normalize by it.
            changes=amplitude*sentinel_shape
            passed=bool(np.max(abs(changes))<=.02)
            maximum=float(amplitude*shape.max()*temporal.max())
            witness=int(np.argmax(shape));key=f'{cid}_{name}'
            arrays[key+'_points']=z['final_points'];arrays[key+'_day_slope']=z['final_day'];arrays[key+'_shape']=shape;arrays[key+'_temporal']=temporal
            arrays[key+'_sentinel_positions']=sites;arrays[key+'_reference_position']=reference;arrays[key+'_sentinel_ratio']=1+changes
            rows.append(dict(case=cid,example=name,amplitude=amplitude,near_ratio_change_pct=100*changes[0],far_ratio_change_pct=100*changes[1],
                endpoint_reference_drift_pct=0,interior_reference_residual_pct=0,session_screen_pass=int(passed),maximum_task_shape_change_pct=100*maximum,
                declared_full_plane_condition_met=int(maximum<=.05+1e-12),witness_point=witness+1,witness_x_m=float(z['final_points'][witness,0]),witness_y_m=float(z['final_points'][witness,1]),
                witness_time_s=int(np.argmax(temporal)),raw_signal_role='ideal expected signals; noise-free counterexample to logical implication',
                eligibility_role='screen pass does not imply independent full-plane condition'))
    arrays['time_seconds']=t;arrays['provenance']=c.PROV
    np.savez_compressed(D/'sentinel_full_plane_counterexample_arrays.npz',**arrays)
    c.write('sentinel_full_plane_counterexamples.csv',rows)
    c.save('sequential_error_domain.json',dict(provenance=c.PROV,time_term=.07,global_shape_term=.05,combined_term=.125,
        temporal_role='conditional 7% full sampling interval bound; observed reference checks alone do not establish unobserved evolution',
        spatial_role='independent condition |rho(p,t)|<=0.05 at every evaluated task point and sampling time; not inferred from two sentinels',
        scalar_reference='fixed near-window workplane location',sentinel_use='screen obvious departures; do not grant full-plane validity',
        allowed_nominal_family='fixed separable affine daylight fields; full-plane bound holds with rho=0',
        bounded_extension='any nonnegative daylight field within explicitly assumed 7% time and 5% full-point multiplicative bounds',
        outside_extension='hidden spatial or recovering bumps above 5%; excluded from conditional outer-bound interpretation even if screen passes',
        bump_definition='product of squared normalized distances to near sentinel, far sentinel and reference, normalized by maximum over 208 points',
        false_inference_counterexamples=6))
    assert sum(r['session_screen_pass'] and not r['declared_full_plane_condition_met'] for r in rows)==6
    return rows

def context_summary():
    raw=c.read(D/'reading_quality_event_context.csv');rows=[]
    for cid in sorted({r['case'] for r in raw}):
        for context in ['freely_controllable','manual_or_hold','input_invalid']:
            rr=[r for r in raw if r['case']==cid and r['context']==context]
            rows.append(dict(case=cid,context=context,input_days=len(rr),Jauto_difference_contribution_pp=float(np.mean([float(r['Jauto_difference_contribution_pp']) for r in rr])),
                reference_eligible_zone_frames=sum(int(r['eligible_zone_frames']) for r in rr),changed_state_frames=sum(int(r['changed_state_frames']) for r in rr)))
        expected=next(r for r in c.read(D/'commissioning_decision_screen.csv') if r['case']==cid)
        assert abs(sum(r['Jauto_difference_contribution_pp'] for r in rows if r['case']==cid)-float(expected['Jauto_difference_pp']))<1e-9
    c.write('reading_quality_event_context_summary.csv',rows)
    return rows

def model_and_labour_definitions():
    c.save('spatial_model_definition.json',dict(provenance=c.PROV,baseline_identity='author-constructed analytic gain plus four-parameter exponential daylight baseline; not a named commercial controller',
        point_source='K_l(p)=4680/pi*h^2/(horizontal_distance_l_squared+h_squared)^2; h=ceiling-0.75',
        circuit_prior='H0_j(p)=sum of 4 circuit point-source terms plus 0.12 times direct circuit mean at frozen 81-point planning grid',
        window_basis='G(p)=finite-window direct coupling with 16x8 quadrature and source weights; raw basis not reference-normalized; alpha absorbs scale',
        reference_normalization='true d(p)=daylight component at p divided by total daylight at (.5*L/11,W/2,0.75); fitted d need not equal 1 at reference',
        baseline_lamp='Hhat_j=a_j*H0_j; unweighted least squares a_j>=0',
        improved_lamp='Hhat_j=a_j*H0_j+b_j; unweighted nonnegative least squares a_j,b_j>=0',
        night_response='average of first two sessions single-circuit minus all-off reading at each fit point',
        slope_response='unweighted through-origin slope from first four sessions x four measured-reference levels',
        baseline_day='exp(c0+c1*x+c2*ycenter+c3*ycenter^2); unweighted least squares on log measured slope; c unrestricted',
        y_center='y minus maximum planning-grid y / 2 (not width/2)',
        improved_day='alpha*G(p)+beta; unweighted nonnegative least squares alpha,beta>=0',
        estimating_readings_per_point=26,validation_readings_per_point=62,layout_protocol_readings_per_point=88,
        model_roles='numerical radiosity benchmark, fitted spatial basis and fixed independent-position observations have distinct roles'))
    old=c.read(D/'commissioning_labour_budget.csv');rows=[]
    for r in old:
        n=int(r['point_count']);r.pop('fit_point_readings',None)
        rows.append(dict(r,coefficient_fit_readings=26*n,layout_validation_readings=62*n,layout_protocol_point_readings=88*n,
            coefficient_fit_hours=26*n*50/3600,layout_validation_hours=62*n*50/3600,layout_protocol_hours=88*n*50/3600,
            cost_role='complete layout-specific fitting and validation protocol; shared qualification separately counted'))
    c.write('commissioning_labour_budget.csv',rows)
    scope=[]
    for r in c.read(D/'commissioning_labour_scope.csv'):
        n=int(r['n_points']);r['layout_protocol_readings']=r.pop('fit_readings',r.get('layout_protocol_readings',88*n))
        r['coefficient_fit_readings']=26*n;r['layout_validation_readings']=62*n
        scope.append(r)
    c.write('commissioning_labour_scope.csv',scope)
    base=json.loads((D/'paper_values.json').read_text('utf8'));base['labour']=rows
    base['validation_evidence']=dict(current_results='analysis_results',reference='camera_reference',paired_grid='208 common independent positions',shape_condition='independent full-plane 5% assumption; not inferred from sentinel passing')
    c.save('paper_values.json',base)

def main():
    D.mkdir(exist_ok=True);qualities=terminal_reconstruction_and_reading();qualification=qualification_examples();sentinels=sentinel_counterexamples();contexts=context_summary();model_and_labour_definitions()
    paired=c.read(D/'paired_grid_validation.csv');point=c.read(D/'paired_point_endpoint_validation.csv');unc=c.read(D/'commissioning_decision_uncertainty.csv')
    c.save('validation_evidence_summary.json',dict(provenance=c.PROV,terminal_roles=json.loads((D/'terminal_evaluation_roles.json').read_text('utf8')),
        noisy_reconstruction_conditions=36,noisy_reconstruction_strategy_days=576,
        quality_summary=[r for r in qualities if r['case']=='ALL36'],qualification_examples=qualification,
        sentinel_counterexamples=sentinels,screen_pass_outside_shape_domain=sum(r['session_screen_pass'] and not r['declared_full_plane_condition_met'] for r in sentinels),
        paired_case_days=288,paired_cases=36,nominal_difference_contained_days=sum(int(r['daily_nominal_contained']) for r in paired),
        paired_endpoint_checks=sum(int(r['endpoint_values']) for r in point),paired_point_exceedances=sum(int(r['lower_exceed_values'])+int(r['upper_exceed_values']) for r in point),
        paired_bounded_extension_checks=sum(int(r['extension_endpoint_values']) for r in point),paired_bounded_extension_exceedances=sum(int(r['extended_lower_exceed_values'])+int(r['extended_upper_exceed_values']) for r in point),
        paired_qualified_reference_outages=sum(int(r['reference_invalid_exposure_frames12'])+int(r['reference_invalid_exposure_frames24']) for r in paired),
        R30_context=[r for r in contexts if r['case']=='R30'],nominal_candidates=[r for r in unc if 'gain_candidate' in r['decision']],
        source_scope='finite nominal fields and explicitly bounded extension; no sentinel-only full-plane certification'))
    print('study validation evidence complete',flush=True)

if __name__=='__main__':main()
