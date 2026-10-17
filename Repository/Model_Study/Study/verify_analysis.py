"""Numerical invariants for decisions, intervals, denominators and counts."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import analysis_core as c
from quality_analysis import qualified_table,automatic_reading
from analysis_engine import fast_engine_trace
W=Path(__file__).resolve().parent;D=W/'analysis_results'
def main():
    checks=[]
    def check(name,value):
        assert value,name
        checks.append(dict(check=name,passed=True))
    main=c.read(D/'current_control_daily.csv');s8=c.read(D/'S8_current_reference_daily.csv');spatial=c.read(D/'spatial_metrics.csv')
    check('principal_scale',len(main)==10368 and len(s8)==3456 and len(spatial)==864)
    for r in main:
        checkval=abs(float(r['lighting_kWh'])+.252-float(r['total_incremental_kWh']))<1e-9 and int(r['manual_conflicts'])==0
        assert checkval,'power or manual conflict'
    check('power_and_manual',True)
    camera=c.read(D/'camera_interval_summary.csv');audit=c.read(D/'qualified_control_replay_audit.csv')
    check('qualified_mixed_bilateral_intervals',all(int(r['both_tails_within_margin']) and int(r['qualified'])==int(r['interval_contained']) for r in camera))
    check('runtime_interval_contains_declared_D',all(int(r['qualified_seconds'])==int(r['contained_seconds']) for r in audit))
    check('all_control_and_S8_trace_consistency',all(int(r['changed_frames'])==0 for r in audit))
    for cid in [f'R{i:02}' for i in range(1,37)]:
        table=np.load(D/(cid+'_lookups.npz'));check('main_S8_'+cid,np.array_equal(table['depth_12_PU'],table['currentPU3']))
    dec=c.read(D/'commissioning_decision_screen.csv');unc=c.read(D/'commissioning_decision_uncertainty.csv')
    check('complete_unique_scene_classification',len(dec)==36 and len({r['case'] for r in dec})==36)
    check('paired_outer_bounds',all(float(r['paired_lower_delta_J_pp'])<=float(r['paired_upper_delta_J_pp'])+1e-10 for r in unc))
    ex=c.read(D/'decision_examples_summary.csv');multi=next(r for r in ex if r['example']=='multiple_feasible_supersets')
    check('superset_power_code_tie_rule',multi['constrained_union']=='3' and multi['DZU']=='11')
    # A reference-domain outage must freeze automatic output, while a local
    # manual command is honoured even during the outage.
    det=np.ones((200,4),np.int8);valid=np.ones(200,np.int8);tab=np.full((40,16),-1,np.int16)
    lookup=np.full((3,16,16,16),15,np.int8);manual=np.array([[10,0,1,150]],np.int64)
    ids,viol,bins,tele=fast_engine_trace(det,valid,tab,lookup,manual)
    check('invalid_reference_local_manual_and_hold',np.all(ids[:10]==0) and np.all(ids[10:]==8) and viol==0 and np.all(tele[:,4]==0))
    budget=c.read(D/'commissioning_labour_scope.csv');x=[r for r in budget if r['workflow']=='research_dense_qualification' and r['n_points']=='12'][0]
    check('research_labour_includes_dense_grid',abs(float(x['spatial_hours'])-200)<1e-10 and int(x['research_acceptance_readings'])==9984)
    components=c.read(D/'energy_fixed_bound_components.csv');check('energy_bound_sum',abs(sum(float(r['value_kWh']) for r in components[:-1])-float(components[-1]['value_kWh']))<1e-12)
    st=c.read(D/'sequential_session_qualification.csv');check('interior_and_shape_failures_detected',all(r['session_screen_pass']=='0' for r in st if r['kind'] in ['returning_cloud','spatial_shape']))
    sentinel=c.read(D/'sentinel_precision_validation.csv');check('low_light_sentinel_precision',len(sentinel)==384 and all(r['qualified']=='1' for r in sentinel))
    fullref=np.load(D/'sequential_full_reference_traces.npz');check('full_sequential_reference_duration',all(fullref[k].shape==(4,7601) for k in fullref.files if k!='provenance'))
    # Shared actual states, not independent absolute quality intervals, determine
    # paired zeros. A frame count is never treated as an independent sample size.
    allpairs=c.read(D/'paired_quality_envelopes.csv')
    check('identical_states_cancel',all(abs(float(r['paired_lower_delta_J_pp']))<1e-12 and abs(float(r['paired_upper_delta_J_pp']))<1e-12 for r in allpairs if float(r['changed_reading_exposure_pct'])==0))
    grid=c.read(D/'paired_grid_validation.csv');endpoint=c.read(D/'paired_point_endpoint_validation.csv')
    check('common_208_point_paired_endpoint',len(grid)==36 and all(int(r['evaluation_positions'])==208 for r in grid))
    check('all_288_nominal_differences_contained',sum(int(r['daily_nominal_contained']) for r in grid)==288)
    check('affine_endpoint_box_contains_nominal_fields',len(endpoint)==1152 and all(int(r['lower_exceed_values'])==0 and int(r['upper_exceed_values'])==0 for r in endpoint))
    check('bounded_full_plane_extension_contained',all(int(r['extended_lower_exceed_values'])==0 and int(r['extended_upper_exceed_values'])==0 for r in endpoint))
    details=c.read(D/'paired_bound_metric_details.csv')
    invalid=[r for r in details if r['reference_role']=='invalid_full_declared_domain']
    check('invalid_reference_uses_full_declared_domain',len(invalid)==36*2*16*4 and all(float(r['daylight_lower_lx'])==0 and float(r['daylight_upper_lx'])==1260 for r in invalid))
    check('same_state_contribution_exact_zero',all(int(r['same_state_lower_contribution'])==0 and int(r['same_state_upper_contribution'])==0 for r in allpairs))
    reading=c.read(D/'all_reading_quality_daily.csv');summ=c.read(D/'all_reading_quality_summary.csv')
    check('complete_all_reading_results',len(reading)==576 and len(summ)==74)
    check('all_reading_contributions_close',all(abs(float(r['Jauto_contribution_pct'])+float(r['Jmanual_contribution_pct'])-float(r['J_pct']))<1e-9 for r in reading))
    check('automatic_exposure_conversion',all(abs(float(r['Jauto_pct'])*float(r['auto_reading_zone_seconds'])/float(r['reading_zone_seconds'])-float(r['Jauto_contribution_pct']))<1e-9 for r in reading))
    examples=c.read(D/'terminal_qualification_examples.csv')
    check('positive_and_negative_qualification_witnesses',len(examples)==2 and {r['research_qualified'] for r in examples}=={'0','1'} and any(int(r['false_accept_candidates'])>0 for r in examples))
    counter=c.read(D/'sentinel_full_plane_counterexamples.csv');domain=json.loads((D/'sequential_error_domain.json').read_text('utf8'))
    check('sentinel_passing_does_not_certify_plane',len(counter)==12 and sum(int(r['session_screen_pass'])==1 and int(r['declared_full_plane_condition_met'])==0 for r in counter)==6)
    check('global_shape_bound_is_explicit_independent_condition','not inferred' in domain['spatial_role'])
    model=json.loads((D/'spatial_model_definition.json').read_text('utf8'));lab=c.read(D/'commissioning_labour_budget.csv')
    check('model_observation_roles_26_plus_62',model['estimating_readings_per_point']==26 and model['validation_readings_per_point']==62)
    check('labour_roles_and_totals_match',all(int(r['coefficient_fit_readings'])+int(r['layout_validation_readings'])==int(r['layout_protocol_point_readings'])==88*int(r['point_count']) for r in lab))
    check('terminal_reconstruction_retains_raw_observations',len(c.read(D/'terminal_reconstruction_diagnostics.csv'))==36 and all(np.load(D/f'R{i:02}_terminal_reconstruction.npz')['observed_night_by_state'].shape==(16,208) for i in range(1,37)))
    context=c.read(D/'reading_quality_event_context_summary.csv')
    check('candidate_contexts_use_common_denominator',len(context)==21 and all(abs(sum(float(r['Jauto_difference_contribution_pp']) for r in context if r['case']==cid)-float(next(r['Jauto_difference_pp'] for r in dec if r['case']==cid)))<1e-9 for cid in sorted({r['case'] for r in context})))
    actions=c.read(D/'action_health_denominator_audit.csv')
    check('action_denominator_includes_camera_health',len(actions)==288 and all(int(r['communication_valid_free_frames'])-int(r['camera_invalid_frames_removed'])==int(r['effective_healthy_free_frames']) for r in actions))
    report=dict(status='passed',checks=checks)
    (D/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('study invariants passed',len(checks),flush=True)
if __name__=='__main__':main()
