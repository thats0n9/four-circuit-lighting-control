"""Check observation identities, denominators and a fresh primary reanalysis."""
from pathlib import Path
import csv
import json
import subprocess
import sys
import tempfile
import numpy as np

HERE=Path(__file__).resolve().parent


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def equivalent(first,second):
    if isinstance(first,dict):return first.keys()==second.keys() and all(equivalent(first[key],second[key]) for key in first)
    if isinstance(first,list):return len(first)==len(second) and all(equivalent(left,right) for left,right in zip(first,second))
    if isinstance(first,(int,float)) and not isinstance(first,bool):return bool(np.isclose(first,second,rtol=1e-10,atol=1e-10))
    if isinstance(first,str):
        try:return bool(np.isclose(float(first),float(second),rtol=1e-10,atol=1e-10))
        except (ValueError,TypeError):return first==second
    return first==second


def main():
    checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(dict(name=name,passed=True))
    data=HERE/'data';results=HERE/'results'
    forbidden={'true_H','true_day','final_H','final_day','pred_H','pred_day','prior','direct_day'}
    for name in ['commissioning.npz','terminal.npz','protection.npz','geometry.npz']:
        with np.load(data/name,allow_pickle=False) as archive:check('no_benchmark_arrays_in_'+name,not forbidden.intersection(archive.files))
    with np.load(data/'terminal.npz',allow_pickle=False) as archive:
        terminal={key:archive[key].copy() for key in archive.files}
    with np.load(data/'geometry.npz',allow_pickle=False) as archive:points=archive['points'].copy()
    with np.load(data/'protection.npz',allow_pickle=False) as archive:protection=archive['points'].copy()
    check('terminal_reading_dimensions',terminal['validation_readings'].shape==(3,16,208))
    check('four_equal_terminal_zones',np.array_equal(np.bincount(terminal['zones']),[52,52,52,52]))
    check('independent_terminal_positions',np.min(np.linalg.norm(terminal['points'][:,None]-points[None],axis=2))>1e-6)
    check('independent_protection_positions',np.min(np.linalg.norm(protection[:,None]-points[None],axis=2))>1e-6)
    check('protection_terminal_separation',np.min(np.linalg.norm(protection[:,None]-terminal['points'][None],axis=2))>1e-6)
    selected=read(results/'selected_positions.csv')
    for layout in ['Depth-stratified','Distributed','Window-clustered']:
        sets={count:{row['point_id'] for row in selected if row['layout']==layout and int(row['n_points'])==count} for count in [8,12,16,24]}
        check('nested_positions_'+layout,all(len(sets[count])==count for count in sets) and sets[8]<sets[12]<sets[16]<sets[24])
    metrics=read(results/'observed_snapshot_metrics.csv');check('snapshot_denominator',len(metrics)==192)
    spatial=read(results/'spatial_metrics.csv');check('twenty_four_spatial_fits',len(spatial)==24)
    check('all_layout_validation_readings_accounted',all(int(row['layout_validation_readings'])==62*int(row['n_points']) for row in spatial))
    expected_pass=0
    for row in metrics:
        level=[0,200,500].index(int(row['background_lx']));state=int(row['state']);zone=int(row['zone'])-1
        values=terminal['validation_readings'][level,state,terminal['zones']==zone]
        actual=int(values.mean()>=500 and values.min()>=300 and values.min()/max(values.mean(),1)>=.6)
        check('snapshot_'+str(level)+'_'+str(state)+'_'+str(zone),actual==int(row['task_pass']))
        expected_pass+=actual
    check('observed_pass_total',expected_pass==int(spatial[0]['observed_pass_combinations']))
    raw=read(data/'metering/assigned_days.csv');daily=read(results/'metered_daily.csv')
    check('forty_eight_days',len(raw)==len(daily)==48)
    for before,after in zip(raw,daily):
        check('register_difference_day_'+before['day'],abs(float(before['register_end_kWh'])-float(before['register_start_kWh'])-float(after['direct_kWh']))<1e-12)
    check('twelve_days_each_policy',all(sum(row['policy']==policy for row in daily)==12 for policy in ['FZ12','P8','P12','P24']))
    check('all_days_meter_eligible',all(row['actual_day_eligible']=='1' for row in daily))
    eligibility=read(results/'metering_statistics/daily_eligibility.csv')
    check('recorded_output_eligibility_schema',all('execution_eligible' in row and 'replay_eligible' not in row for row in eligibility))
    check('recorded_policy_labels_retained',all(row['policy']==original['policy'] and row['applied_policy']==original['applied_policy'] for row,original in zip(eligibility,raw)))
    check('record_group_traceability',all(row['provenance']==original['case'] for row,original in zip(eligibility,raw)))
    sys.path.insert(0,str(HERE.parent/'documentation'))
    from build_data_dictionary import build_rows,verify_dictionary
    dictionary=read(HERE.parent/'documentation/primary_data_dictionary.csv')
    dictionary_report=verify_dictionary(HERE.parent,dictionary)
    check('primary_dictionary_fields_resolve',dictionary_report['status']=='passed')
    check('primary_dictionary_reproducible',dictionary==build_rows(HERE.parent))
    check('fitted_parameter_dictionary_paths',any(row['file'].endswith('fitted_parameters.json') and row['field']=='$[*]["lamp_coefficients"]' for row in dictionary))
    indexed_files={row['file'] for row in dictionary}
    check('primary_dictionary_covers_results',all(path.relative_to(HERE.parent).as_posix() in indexed_files for path in results.rglob('*') if path.is_file()))
    check('shared_register_values',all(float(raw[index]['register_start_kWh'])==float(raw[index-1]['register_end_kWh']) for index in range(1,48)))
    details=read(results/'snapshot_decision_details.csv')
    check('snapshot_detail_denominator',len(details)==24*192)
    classifications=read(results/'snapshot_classification.csv')
    check('classification_denominator',len(classifications)==48)
    for row in classifications:
        group=[item for item in details if all(item[key]==row[key] for key in ['layout','n_points','model'])]
        field='unprotected_pass' if row['stage']=='Unprotected' else 'protected_pass'
        truth=np.array([item['observed_pass']=='1' for item in group]);predicted=np.array([item[field]=='1' for item in group])
        actual=[int((truth&predicted).sum()),int((~truth&predicted).sum()),int((~truth&~predicted).sum()),int((truth&~predicted).sum())]
        check('classification_'+row['layout']+'_'+row['n_points']+'_'+row['model']+'_'+row['stage'],actual==[int(row[key]) for key in ['true_positive','false_positive','true_negative','false_negative']])
    requests=read(results/'snapshot_request_candidates.csv')
    check('request_candidate_denominator',len(requests)==24*45)
    grouped_details={(item['layout'],item['n_points'],item['model'],item['background_lx'],item['state'],item['zone']):item for item in details}
    for index,row in enumerate(requests):
        zones=row['requested_zones'].split(';')
        for stage,field in [('observed','observed_pass'),('unprotected','unprotected_pass'),('protected','protected_pass')]:
            states=[state for state in range(16) if all(grouped_details[row['layout'],row['n_points'],row['model'],row['background_lx'],str(state),zone][field]=='1' for zone in zones)]
            check('request_set_'+str(index)+'_'+stage,';'.join(map(str,states))==row[stage+'_candidate_states'] and len(states)==int(row[stage+'_candidate_count']))
    mechanism=json.loads((results/'snapshot_mechanism_summary.json').read_text('utf-8'))
    check('protected_sets_identical',mechanism['point_count_request_comparison']['protected']['identical_candidate_sets']==45)
    check('four_static_rank_changes',mechanism['point_count_request_comparison']['unprotected']['changed_minimum_power_state_among_common']==4)
    check('protected_false_rejection_cost',all(row['false_negative']==7 and row['true_positive']==39 for row in mechanism['classification'] if row['stage']=='Protected'))
    events=read(data/'metering/node_execution_events.csv');execution=read(results/'execution_daily.csv');hours=read(results/'execution_state_hours.csv')
    check('recorded_interval_count',len(read(results/'execution_intervals.csv'))==len(events)==2405)
    power=np.array([float(row['estimated_power_kW']) for row in read(results/'metering_statistics/state_power.csv')])
    for row in execution:
        day=row['day'];actual=next(item for item in raw if item['day']==day)
        selected=sorted([item for item in events if item['day']==day],key=lambda item:float(item['event_seconds']))
        timestamps=np.array([float(item['event_seconds']) for item in selected]+[float(actual['duration_h'])*3600])
        independent=np.zeros(16)
        for position,event in enumerate(selected):independent[int(event['state'])]+=(timestamps[position+1]-timestamps[position])/3600
        published=np.array([float(item['hours']) for item in hours if item['day']==day])
        check('state_hours_'+day,np.allclose(independent,published,rtol=1e-10,atol=1e-10) and abs(independent.sum()-14)<1e-10)
        check('trace_integral_'+day,abs(independent@power-float(row['calibration_integrated_kWh']))<1e-10)
        check('meter_trace_closure_'+day,abs(float(row['metered_kWh'])-float(row['calibration_integrated_kWh'])-float(row['meter_minus_calibration_kWh']))<1e-10 and row['closure_pass']=='1')
    execution_summary=json.loads((results/'execution_mechanism_summary.json').read_text('utf-8'))
    check('execution_energy_identity',abs(execution_summary['P24_minus_P12_metered_kWh']-execution_summary['P24_minus_P12_calibration_integrated_kWh']-execution_summary['P24_minus_P12_residual_kWh'])<1e-10)
    check('trace_not_independent_lamp_feedback',execution_summary['source_state_series_kind']=='applied_output_trace' and execution_summary['independent_continuous_lamp_feedback'] is False)
    with tempfile.TemporaryDirectory(prefix='bsert_observation_check_') as temporary:
        generated=Path(temporary)/'reanalysis'
        subprocess.run([sys.executable,'-B',str(HERE/'observed_analysis.py'),'--destination',str(generated)],check=True,stdout=subprocess.DEVNULL)
        for path in sorted(results.rglob('*')):
            if not path.is_file():continue
            target=generated/path.relative_to(results)
            check('reproduced_'+path.relative_to(results).as_posix(),target.is_file())
            if path.suffix=='.csv':check('equivalent_'+path.relative_to(results).as_posix(),equivalent(read(path),read(target)))
            elif path.suffix=='.json':check('equivalent_'+path.relative_to(results).as_posix(),equivalent(json.loads(path.read_text('utf-8-sig')),json.loads(target.read_text('utf-8-sig'))))
        assert generated.resolve().is_relative_to(Path(temporary).resolve())
    report=dict(status='passed',checks_passed=len(checks),checks=checks,primary_reanalysis_repeated=True,model_benchmark_arrays_loaded=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
