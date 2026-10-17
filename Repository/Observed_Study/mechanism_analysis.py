"""Decision and applied-output diagnostics anchored to the fixed observations."""
from pathlib import Path
import json
import numpy as np
from observed_analysis import arrays, basis, metrics, read, write, save, STATES

TARGETS=np.array([500.,300.,.6])
METRICS=['mean_lx','minimum_lx','uniformity']


def metric_values(row):return np.array([float(row[name]) for name in METRICS])


def metric_key(row):return int(row['background_lx']),int(row['state']),int(row['zone'])


def configuration_key(row):return row['layout'],int(row['n_points']),row['model']


def predicted_metrics(parameter,points,candidate_points,zones):
    circuit,window=basis(points,candidate_points)
    lamp=np.array(parameter['lamp_coefficients']);coefficient=np.array(parameter['day_coefficients'])
    if parameter['model']=='Analytic baseline':
        lateral=points[:,1]-candidate_points[:,1].max()/2
        daylight=np.exp(np.column_stack([np.ones(len(points)),points[:,0],lateral,lateral*lateral])@coefficient)
    else:daylight=window*coefficient[0]+coefficient[1]
    prediction=STATES@(circuit*lamp[:,0]+lamp[:,1]).T+np.array([0,200,500])[:,None,None]*daylight[None,None]
    return metrics(prediction,zones)


def snapshot_diagnostics(data,destination):
    terminal=arrays(data/'terminal.npz',['points','zones','validation_readings'])
    candidate=arrays(data/'geometry.npz',['points'])['points']
    observed={metric_key(row):row for row in metrics(terminal['validation_readings'],terminal['zones'])}
    allowances=read(destination/'snapshot_allowances.csv')
    allowance_map={(configuration_key(row),metric_key(row)):row for row in allowances}
    archived={(configuration_key(row),metric_key(row)):row for row in read(destination/'prediction_snapshot_acceptance.csv')}
    power={int(row['state']):float(row['estimated_power_kW']) for row in read(destination/'metering_statistics/state_power.csv')}
    parameters=json.loads((destination/'fitted_parameters.json').read_text('utf-8'))
    details=[];classification=[];false_cases=[];request_rows=[];configuration_summaries=[]
    configurations={}
    for parameter in parameters:
        configuration=configuration_key(parameter)
        label=dict(layout=configuration[0],n_points=configuration[1],model=configuration[2])
        prediction=predicted_metrics(parameter,terminal['points'],candidate,terminal['zones'])
        grouped={}
        for row in prediction:
            key=metric_key(row);truth=observed[key];allowance=allowance_map[configuration,key]
            predicted=metric_values(row);actual=metric_values(truth)
            margin=np.array([float(allowance[name]) for name in ['mean_allowance_lx','minimum_allowance_lx','uniformity_allowance']])
            raw_slack=predicted/TARGETS-1;protected_slack=(predicted-margin)/TARGETS-1;actual_slack=actual/TARGETS-1
            raw_pass=bool(np.all(predicted>=TARGETS));protected_pass=bool(np.all(predicted-margin>=TARGETS));actual_pass=bool(truth['task_pass'])
            previous=archived[configuration,key]
            assert int(raw_pass)==int(previous['predicted_pass']) and int(protected_pass)==int(previous['protected_pass'])
            detail=dict(**label,background_lx=key[0],state=key[1],zone=key[2],observed_pass=int(actual_pass),unprotected_pass=int(raw_pass),protected_pass=int(protected_pass),observed_normalised_slack=float(actual_slack.min()),unprotected_normalised_slack=float(raw_slack.min()),protected_normalised_slack=float(protected_slack.min()),observed_limiting_metric=METRICS[int(actual_slack.argmin())],unprotected_limiting_metric=METRICS[int(raw_slack.argmin())])
            for index,name in enumerate(METRICS):
                detail['observed_'+name]=float(actual[index]);detail['predicted_'+name]=float(predicted[index]);detail['allowance_'+name]=float(margin[index]);detail['protected_'+name]=float(predicted[index]-margin[index])
            details.append(detail);grouped[key]=detail
            if raw_pass and not actual_pass:
                false_cases.append(dict(**detail,observed_failed_metrics=';'.join(name for index,name in enumerate(METRICS) if actual[index]<TARGETS[index])))
        configurations[configuration]=grouped
        for stage,field in [('Unprotected','unprotected_pass'),('Protected','protected_pass')]:
            true_positive=sum(row[field] and row['observed_pass'] for row in grouped.values())
            false_positive=sum(row[field] and not row['observed_pass'] for row in grouped.values())
            false_negative=sum(not row[field] and row['observed_pass'] for row in grouped.values())
            true_negative=192-true_positive-false_positive-false_negative
            classification.append(dict(**label,stage=stage,true_positive=true_positive,false_positive=false_positive,true_negative=true_negative,false_negative=false_negative,predicted_passes=true_positive+false_positive,observed_passes=46,total=192,precision_pct=100*true_positive/(true_positive+false_positive) if true_positive+false_positive else '',retained_observed_passes_pct=100*true_positive/46))
        counts=dict(**label)
        for level in [0,200,500]:
            for request in range(1,16):
                zones=[index+1 for index,flag in enumerate(STATES[request]) if flag]
                sets={stage:[state for state in range(16) if all(grouped[level,state,zone][field] for zone in zones)] for stage,field in [('Observed','observed_pass'),('Unprotected','unprotected_pass'),('Protected','protected_pass')]}
                item=dict(**label,background_lx=level,request_mask=request,requested_zones=';'.join(str(zone) for zone in zones),requested_zone_count=len(zones))
                for stage,states in sets.items():
                    prefix=stage.lower();selected=min(states,key=lambda state:(power[state],state)) if states else None
                    item[prefix+'_candidate_count']=len(states);item[prefix+'_candidate_states']=';'.join(str(state) for state in states)
                    item[prefix+'_selected_state']='' if selected is None else selected
                    item[prefix+'_selected_calibrated_power_kW']='' if selected is None else power[selected]
                    item[prefix+'_selected_observed_pass']='' if selected is None else int(selected in sets['Observed'])
                request_rows.append(item)
        current=[row for row in request_rows if configuration_key(row)==configuration]
        for stage in ['observed','unprotected','protected']:
            counts[stage+'_feasible_requests']=sum(row[stage+'_candidate_count']>0 for row in current)
            counts[stage+'_unsafe_selected_requests']=sum(row[stage+'_selected_observed_pass']==0 for row in current if row[stage+'_selected_state']!='')
        configuration_summaries.append(counts)
    write(destination/'snapshot_decision_details.csv',details)
    write(destination/'snapshot_classification.csv',classification)
    write(destination/'snapshot_false_acceptances.csv',false_cases)
    write(destination/'snapshot_request_candidates.csv',request_rows)
    write(destination/'snapshot_request_summary.csv',configuration_summaries)
    paired=[]
    first=configurations['Depth-stratified',12,'Improved model'];second=configurations['Depth-stratified',24,'Improved model']
    for key in first:
        before,after=first[key],second[key]
        paired.append(dict(background_lx=key[0],state=key[1],zone=key[2],observed_pass=before['observed_pass'],unprotected12=before['unprotected_pass'],unprotected24=after['unprotected_pass'],protected12=before['protected_pass'],protected24=after['protected_pass'],slack12=before['unprotected_normalised_slack'],slack24=after['unprotected_normalised_slack'],protected_slack12=before['protected_normalised_slack'],protected_slack24=after['protected_normalised_slack']))
    write(destination/'snapshot_point_count_pairs.csv',paired)
    first_requests={(int(row['background_lx']),int(row['request_mask'])):row for row in request_rows if configuration_key(row)==('Depth-stratified',12,'Improved model')}
    second_requests={(int(row['background_lx']),int(row['request_mask'])):row for row in request_rows if configuration_key(row)==('Depth-stratified',24,'Improved model')}
    comparison={}
    for stage in ['unprotected','protected']:
        comparisons=[(first_requests[key],second_requests[key]) for key in first_requests]
        common=[(before,after) for before,after in comparisons if before[stage+'_candidate_count'] and after[stage+'_candidate_count']]
        comparison[stage]=dict(requests=45,identical_candidate_sets=sum(before[stage+'_candidate_states']==after[stage+'_candidate_states'] for before,after in comparisons),common_feasible_requests=len(common),same_minimum_power_state_among_common=sum(before[stage+'_selected_state']==after[stage+'_selected_state'] for before,after in common),changed_minimum_power_state_among_common=sum(before[stage+'_selected_state']!=after[stage+'_selected_state'] for before,after in common),availability_disagreements=sum(bool(before[stage+'_candidate_count'])!=bool(after[stage+'_candidate_count']) for before,after in comparisons))
    selected=[row for row in classification if row['layout']=='Depth-stratified' and row['model']=='Improved model' and row['n_points'] in [12,24]]
    selected_requests=[row for row in configuration_summaries if row['layout']=='Depth-stratified' and row['model']=='Improved model' and row['n_points'] in [12,24]]
    summary=dict(classification=selected,request_summary=selected_requests,point_count_request_comparison=comparison,unprotected_changed_snapshot_labels=sum(row['unprotected12']!=row['unprotected24'] for row in paired),protected_changed_snapshot_labels=sum(row['protected12']!=row['protected24'] for row in paired),scope='Independent observed snapshots; 45 exhaustive request/background combinations, not operating-day samples',selection_rule='Minimum recorded calibration power centre then state code, among snapshot-feasible candidates only; no fallback, manual rule or dynamic policy replay',power_role='Calibrated power used only for static candidate ranking; no daily electricity outcome is inferred')
    save(destination/'snapshot_mechanism_summary.json',summary)
    return summary


def execution_diagnostics(data,destination):
    allocated={int(row['day']):row for row in read(data/'metering/assigned_days.csv')}
    source_events=read(data/'metering/node_execution_events.csv')
    eligibility={int(row['day']):row for row in read(destination/'metering_statistics/daily_eligibility.csv')}
    power=np.array([float(row['estimated_power_kW']) for row in read(destination/'metering_statistics/state_power.csv')])
    settings=json.loads((data/'metering/execution_settings.json').read_text('utf-8'))
    daily=[];intervals=[];state_rows=[]
    for day,allocation in sorted(allocated.items()):
        events=sorted([row for row in source_events if int(row['day'])==day],key=lambda row:float(row['event_seconds']))
        times=np.array([float(row['event_seconds']) for row in events]);states=np.array([int(row['state']) for row in events])
        duration=float(allocation['duration_h']);ends=np.r_[times[1:],duration*3600]
        assert times[0]==0 and np.all(ends>times) and eligibility[day]['log_complete']=='1'
        hours=np.bincount(states,weights=(ends-times)/3600,minlength=16)
        metered=float(allocation['register_end_kWh'])-float(allocation['register_start_kWh'])
        integrated=float(hours@power);residual=metered-integrated
        assert abs(integrated-float(eligibility[day]['integrated_kWh']))<1e-10
        bit_changes=[int(before^after).bit_count() for before,after in zip(states[:-1],states[1:])]
        lit_hours=np.bincount(STATES.sum(axis=1),weights=hours,minlength=5)
        item=dict(day=day,block=int(allocation['block']),policy=allocation['policy'],duration_h=duration,recorded_events=len(events),output_state_changes=int(sum(before!=after for before,after in zip(states[:-1],states[1:]))),individual_circuit_changes=sum(bit_changes),calibration_integrated_kWh=integrated,metered_kWh=metered,meter_minus_calibration_kWh=residual,closure_bound_kWh=float(eligibility[day]['closure_bound_kWh']),closure_pass=int(abs(residual)<=float(eligibility[day]['closure_bound_kWh'])))
        for count in range(5):item[f'{count}_circuits_on_hours']=float(lit_hours[count])
        daily.append(item)
        for start,end,state in zip(times,ends,states):intervals.append(dict(day=day,policy=allocation['policy'],start_s=float(start),end_s=float(end),duration_h=float((end-start)/3600),state=int(state),circuit_bits=''.join(str(bit) for bit in STATES[state]),circuits_on=int(STATES[state].sum())))
        for state in range(16):state_rows.append(dict(day=day,policy=allocation['policy'],state=state,hours=float(hours[state]),fraction_of_day=float(hours[state]/duration)))
    summary=[]
    for policy in ['FZ12','P8','P12','P24']:
        group=[row for row in daily if row['policy']==policy]
        item=dict(policy=policy,days=len(group))
        for field in ['output_state_changes','individual_circuit_changes','calibration_integrated_kWh','metered_kWh','meter_minus_calibration_kWh']+[f'{count}_circuits_on_hours' for count in range(5)]:item['mean_'+field]=float(np.mean([row[field] for row in group]))
        item['closure_pass_days']=sum(row['closure_pass'] for row in group);summary.append(item)
    first=[row for row in daily if row['policy']=='P12'];second=[row for row in daily if row['policy']=='P24']
    mean_hours=lambda policy:np.array([np.mean([row['hours'] for row in state_rows if row['policy']==policy and row['state']==state]) for state in range(16)])
    before=mean_hours('P12');after=mean_hours('P24');delta=after-before
    decomposition=[dict(state=state,circuits_on=int(STATES[state].sum()),P12_mean_hours=float(before[state]),P24_mean_hours=float(after[state]),delta_hours=float(delta[state]),calibrated_power_kW=float(power[state]),calibration_integrated_contribution_kWh=float(delta[state]*power[state])) for state in range(16)]
    observed_delta=float(np.mean([row['metered_kWh'] for row in second])-np.mean([row['metered_kWh'] for row in first]))
    integrated_delta=float(delta@power)
    residual_delta=float(np.mean([row['meter_minus_calibration_kWh'] for row in second])-np.mean([row['meter_minus_calibration_kWh'] for row in first]))
    assert abs(observed_delta-integrated_delta-residual_delta)<1e-10
    report=dict(days=48,intervals=len(intervals),source_state_series_kind=settings['state_series_kind'],independent_continuous_lamp_feedback=settings['independent_continuous_lamp_feedback'],scope='Recorded applied-output trace; distinct operating days, not common-input paired execution',policy_summary=summary,P24_minus_P12_metered_kWh=observed_delta,P24_minus_P12_calibration_integrated_kWh=integrated_delta,P24_minus_P12_residual_kWh=residual_delta,mean_state_distribution_total_variation=float(.5*np.abs(delta/14).sum()),closure_pass_days=sum(row['closure_pass'] for row in daily),total_recorded_output_changes=sum(row['output_state_changes'] for row in daily),total_individual_circuit_changes=sum(row['individual_circuit_changes'] for row in daily),metered_energy_remains_primary=True,trace_integral_role='Model calculation from recorded state durations and calibration power centres; diagnostic only, not an additional meter reading',state_distribution_role='Descriptive distributional distance between different allocated days, not the fraction of matched frames with different outputs')
    write(destination/'execution_intervals.csv',intervals);write(destination/'execution_daily.csv',daily);write(destination/'execution_state_hours.csv',state_rows);write(destination/'execution_policy_summary.csv',summary);write(destination/'execution_contrast_decomposition.csv',decomposition)
    save(destination/'execution_mechanism_summary.json',report)
    return report


def analyse(data,destination):
    spatial=snapshot_diagnostics(Path(data),Path(destination))
    execution=execution_diagnostics(Path(data),Path(destination))
    return dict(snapshot=spatial,execution=execution)
