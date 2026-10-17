"""Fit spatial responses and analyse fixed illuminance and cumulative-meter records."""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import sys
import tempfile
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'lib'))
from metering_analysis import process
from statistical_utils import tcrit

STATES = np.array([[int(char) for char in f'{state:04b}'] for state in range(16)], dtype=np.int8)
LAYOUTS = ['Depth-stratified', 'Distributed', 'Window-clustered']


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, values):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def arrays(path, keys):
    forbidden = {'true_H', 'true_day', 'final_H', 'final_day', 'pred_H', 'pred_day', 'prior', 'direct_day'}
    assert not forbidden.intersection(keys)
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in keys}


def selections(points):
    zones = np.minimum((points[:,1]/2.05).astype(int),3)
    output = {}
    for layout in LAYOUTS:
        orders = []
        for zone in range(4):
            pool = np.flatnonzero(zones==zone).tolist()
            if layout == 'Depth-stratified':
                representative = [min(pool,key=lambda index:float(np.sum((points[index]-[position,(zone+.5)*2.05])**2))) for position in [.5,5.5,10.5]]
                chosen = [representative[0],representative[2],representative[1]]
            else:
                if layout == 'Window-clustered':
                    pool = [index for index in pool if points[index,0]<=3.0]
                centre = np.array([.5 if layout=='Window-clustered' else 5.5,(zone+.5)*2.05])
                chosen = [min(pool,key=lambda index:(float(np.sum((points[index]-centre)**2)),index))]
            while len(chosen)<6:
                remaining = [index for index in pool if index not in chosen]
                chosen.append(min(remaining,key=lambda index:(-float(np.min(np.sum(((points[index]-points[chosen])/np.array([11.,2.05]))**2,axis=1))),index)))
            orders.append(chosen)
        for count in [8,12,16,24]:
            output[layout,count] = np.array([index for order in orders for index in order[:count//4]],int)
    return output


def basis(points, candidate_points):
    height = 2.25
    lamps = np.array([[(position+.5)*11/4,(zone+.5)*8.2/4] for zone in range(4) for position in range(4)])
    def direct_at(locations):
        distance = np.sum((locations[:,None]-lamps[None])**2,axis=2)+height*height
        direct = 4680/math.pi*height*height/distance**2
        return np.stack([direct[:,zone*4:(zone+1)*4].sum(1) for zone in range(4)],axis=1)
    circuit = direct_at(points)+.12*direct_at(candidate_points).mean(axis=0)
    lower,upper=1.0,1.0+(3.0-1.3)/2
    source = np.array([[.001,.15*8.2+(position+.5)/16*(.7*8.2),lower+(vertical+.5)/8*(upper-lower)] for position in range(16) for vertical in range(8)])
    target = np.column_stack([points,np.full(len(points),.75)])
    displacement = source[None]-target[:,None]
    distance_squared = np.sum(displacement*displacement,axis=2)
    area = (.7*8.2)*(upper-lower)/128
    daylight = np.sum(np.maximum(displacement[:,:,2],0)*np.maximum(-displacement[:,:,0],0)/(math.pi*distance_squared**2)*area,axis=1)
    return circuit,daylight


def nnls_two(design,response):
    unconstrained = np.linalg.lstsq(design,response,rcond=None)[0]
    candidates=[np.maximum(unconstrained,0),np.array([max(0,float(design[:,0]@response/(design[:,0]@design[:,0]))),0.]),np.array([0.,max(0,float(design[:,1]@response/(design[:,1]@design[:,1])))]),np.zeros(2)]
    return min(candidates,key=lambda coefficient:float(np.sum((design@coefficient-response)**2)))


def metrics(values,zones):
    output=[]
    for level in range(3):
        for state in range(16):
            for zone in range(4):
                group=values[level,state,zones==zone]
                average=float(group.mean());minimum=float(group.min());uniformity=minimum/max(average,1.)
                output.append(dict(background_lx=[0,200,500][level],state=state,zone=zone+1,mean_lx=average,minimum_lx=minimum,uniformity=uniformity,task_pass=int(average>=500 and minimum>=300 and uniformity>=.6)))
    return output


def spatial(data,destination):
    pool=arrays(data/'commissioning.npz',['night','day','reference'])
    geometry=arrays(data/'geometry.npz',['points'])
    terminal=arrays(data/'terminal.npz',['points','zones','validation_readings'])
    protection=arrays(data/'protection.npz',['points','zones','night_readings','day_readings','reference_readings','day_levels_lx'])
    points=geometry['points'];endpoints=terminal['points'];zones=terminal['zones'];observed=terminal['validation_readings']
    prior,direct=basis(points,points);final_prior,final_direct=basis(endpoints,points)
    protection_prior,protection_direct=basis(protection['points'],points)
    protection_reference=protection['reference_readings'].ravel()
    protection_response=protection['day_readings'].reshape(8,69)
    protection_slope=(protection_reference[:,None]*protection_response).sum(axis=0)/(protection_reference@protection_reference)
    protection_field=protection['night_readings'].mean(axis=0)[None]+np.array([0,200,500])[:,None,None]*protection_slope[None,None]
    protection_metrics=metrics(protection_field,protection['zones'])
    selected=selections(points)
    rows=[];parameters=[];selection_rows=[];candidate_rows=[];allowance_rows=[]
    observed_metrics=metrics(observed,zones)
    flags=np.array([row['task_pass'] for row in observed_metrics],bool)
    day_reference=pool['reference'][:4].ravel()
    for (layout,count),indices in selected.items():
        for index in indices:
            selection_rows.append(dict(layout=layout,n_points=count,point_id=int(index)+1,x_m=float(points[index,0]),y_m=float(points[index,1])))
        response=np.column_stack([(pool['night'][:2,state,indices]-pool['night'][:2,0,indices]).mean(0) for state in [8,4,2,1]])
        daylight_values=pool['day'][:4,:,indices].reshape(16,len(indices))
        slope=(day_reference[:,None]*daylight_values).sum(axis=0)/(day_reference@day_reference)
        for method in ['Analytic baseline','Improved model']:
            if method=='Analytic baseline':
                gain=np.maximum(0,np.sum(response*prior[indices],axis=0)/np.sum(prior[indices]**2,axis=0))
                lamp=np.column_stack([gain,np.zeros(4)])
                centre=points[:,1].max()/2
                design=np.column_stack([np.ones(len(indices)),points[indices,0],points[indices,1]-centre,(points[indices,1]-centre)**2])
                day_coefficient=np.linalg.lstsq(design,np.log(slope),rcond=None)[0]
                final_y=endpoints[:,1]-centre
                predicted_day=np.exp(np.column_stack([np.ones(len(endpoints)),endpoints[:,0],final_y,final_y**2])@day_coefficient)
                candidate_y=points[:,1]-centre
                candidate_day=np.exp(np.column_stack([np.ones(81),points[:,0],candidate_y,candidate_y**2])@day_coefficient)
                protection_y=protection['points'][:,1]-centre
                protection_day=np.exp(np.column_stack([np.ones(69),protection['points'][:,0],protection_y,protection_y**2])@day_coefficient)
            else:
                lamp=np.array([nnls_two(np.column_stack([prior[indices,zone],np.ones(len(indices))]),response[:,zone]) for zone in range(4)])
                day_coefficient=nnls_two(np.column_stack([direct[indices],np.ones(len(indices))]),slope)
                predicted_day=final_direct*day_coefficient[0]+day_coefficient[1]
                candidate_day=direct*day_coefficient[0]+day_coefficient[1]
                protection_day=protection_direct*day_coefficient[0]+day_coefficient[1]
            prediction=STATES@(final_prior*lamp[:,0]+lamp[:,1]).T+np.array([0,200,500])[:,None,None]*predicted_day[None,None]
            errors=prediction-observed
            candidate_night=STATES@(prior[indices]*lamp[:,0]+lamp[:,1]).T
            recorded_night=pool['night'][:,:,indices]
            mixed_states=[state for state in range(16) if state not in [0,1,2,4,8]]
            first_mixed=(candidate_night[None,mixed_states,:]-recorded_night[:2,mixed_states,:]).ravel()
            held_night=(candidate_night[None]-recorded_night[2:]).ravel()
            held_day=(pool['reference'][4:,:,None]*candidate_day[indices]-pool['day'][4:,:,indices]).ravel()
            layout_validation=np.concatenate([first_mixed,held_night,held_day])
            assert len(layout_validation)==62*count
            planned=metrics(prediction,zones)
            planned_flags=np.array([row['task_pass'] for row in planned],bool)
            protection_prediction=STATES@(protection_prior*lamp[:,0]+lamp[:,1]).T+np.array([0,200,500])[:,None,None]*protection_day[None,None]
            protection_planned=metrics(protection_prediction,protection['zones'])
            protected_flags=[]
            for planned_row,reference_row,terminal_row in zip(protection_planned,protection_metrics,planned):
                mean_allowance=max(0,planned_row['mean_lx']-reference_row['mean_lx'])+.03*reference_row['mean_lx']+1
                minimum_allowance=max(0,planned_row['minimum_lx']-reference_row['minimum_lx'])+.03*reference_row['minimum_lx']+1
                uniformity_allowance=max(0,planned_row['uniformity']-reference_row['uniformity'])+.01
                protected_flags.append(terminal_row['mean_lx']-mean_allowance>=500 and terminal_row['minimum_lx']-minimum_allowance>=300 and terminal_row['uniformity']-uniformity_allowance>=.6)
                allowance_rows.append(dict(layout=layout,n_points=count,model=method,background_lx=planned_row['background_lx'],state=planned_row['state'],zone=planned_row['zone'],mean_allowance_lx=mean_allowance,minimum_allowance_lx=minimum_allowance,uniformity_allowance=uniformity_allowance))
            protected_flags=np.array(protected_flags,bool)
            accepted=int(planned_flags.sum());false_accept=int((planned_flags&~flags).sum())
            rows.append(dict(layout=layout,n_points=count,model=method,RMSE_lx=float(np.sqrt(np.mean(errors**2))),MAE_lx=float(abs(errors).mean()),bias_lx=float(errors.mean()),night_RMSE_lx=float(np.sqrt(np.mean(errors[0]**2))),layout_validation_readings=len(layout_validation),layout_validation_RMSE_lx=float(np.sqrt(np.mean(layout_validation**2))),observed_readings=observed.size,planned_pass_combinations=accepted,observed_false_acceptances=false_accept,false_acceptance_pct=100*false_accept/max(accepted,1),protected_pass_combinations=int(protected_flags.sum()),protected_false_acceptances=int((protected_flags&~flags).sum()),observed_pass_combinations=int(flags.sum()),denominator_combinations=192))
            for row,actual,protected in zip(planned,observed_metrics,protected_flags):
                candidate_rows.append(dict(layout=layout,n_points=count,model=method,background_lx=row['background_lx'],state=row['state'],zone=row['zone'],predicted_pass=row['task_pass'],protected_pass=int(protected),observed_pass=actual['task_pass'],false_accept=int(row['task_pass'] and not actual['task_pass']),protected_false_accept=int(protected and not actual['task_pass'])))
            parameters.append(dict(layout=layout,n_points=count,model=method,lamp_coefficients=lamp.tolist(),day_coefficients=day_coefficient.tolist()))
    write(destination/'spatial_metrics.csv',rows)
    write(destination/'selected_positions.csv',selection_rows)
    write(destination/'observed_snapshot_metrics.csv',observed_metrics)
    write(destination/'prediction_snapshot_acceptance.csv',candidate_rows)
    write(destination/'snapshot_allowances.csv',allowance_rows)
    save(destination/'fitted_parameters.json',parameters)
    snapshots=[]
    for level in [0,200,500]:
        for state in range(16):
            group=[row for row in observed_metrics if row['background_lx']==level and row['state']==state]
            snapshots.append(dict(background_lx=level,state=state,passing_zones=sum(row['task_pass'] for row in group),all_zones_pass=int(all(row['task_pass'] for row in group))))
    write(destination/'observed_state_availability.csv',snapshots)
    return rows,snapshots


def energy(data,destination):
    with tempfile.TemporaryDirectory(prefix='metering_',dir=destination) as temporary:
        temporary=Path(temporary).resolve()
        assert temporary.is_relative_to(destination.resolve())
        shutil.copytree(data/'metering',temporary/'records')
        analysed,models,counts=process(temporary/'records')
        shutil.copytree(temporary/'records/processed',destination/'metering_statistics')
        assert temporary.is_relative_to(destination.resolve())
    daily=[]
    for row in analysed:
        daily.append({key:row[key] for key in ['day','block','position','policy','direct_kWh','actual_day_eligible','log_complete','occupied_zone_hours','daylight_reference_lxh']})
    write(destination/'metered_daily.csv',daily)
    summary=[]
    for policy in ['FZ12','P8','P12','P24']:
        values=np.array([row['direct_kWh'] for row in daily if row['policy']==policy and row['actual_day_eligible']])
        summary.append(dict(policy=policy,n_days=len(values),mean_kWh=float(values.mean()),SD_kWh=float(values.std(ddof=1)),sum_kWh=float(values.sum()),min_kWh=float(values.min()),max_kWh=float(values.max())))
    write(destination/'metered_policy_summary.csv',summary)
    paired=[]
    for block in range(1,13):
        group={row['policy']:row for row in daily if row['block']==block}
        paired.append(dict(block=block,P12_kWh=group['P12']['direct_kWh'],P24_kWh=group['P24']['direct_kWh'],delta_kWh=group['P24']['direct_kWh']-group['P12']['direct_kWh']))
    write(destination/'metered_block_contrasts.csv',paired)
    differences=np.array([row['delta_kWh'] for row in paired]);mean=float(differences.mean());standard_error=float(differences.std(ddof=1)/math.sqrt(12));halfwidth=tcrit(12)*standard_error
    sensitivity=dict(estimate_kWh=mean,SE_kWh=standard_error,df=11,CI_low_kWh=mean-halfwidth,CI_high_kWh=mean+halfwidth,unit='one contrast per four-day block',allocation='fixed within-block order, not randomised')
    contrasts=[]
    for name in ['primary','primary_covariate_sensitivity','secondary_8vs12','secondary_control']:
        row=models[name]
        contrasts.append(dict(analysis=name,estimate_kWh=row['estimate_kWh'],CI_low_kWh=row['CI_low_kWh'],CI_high_kWh=row['CI_high_kWh'],fixed_bound_kWh=row['fixed_envelope_kWh'],expanded_low_kWh=row['expanded_interval_low_kWh'],expanded_high_kWh=row['expanded_interval_high_kWh'],n_days=row['n'],df=row['df']))
    write(destination/'metered_contrasts.csv',contrasts)
    save(destination/'block_pair_sensitivity.json',sensitivity)
    save(destination/'metered_models.json',models)
    return models,summary,sensitivity,counts


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=HERE/'data')
    parser.add_argument('--destination',type=Path,required=True)
    arguments=parser.parse_args();destination=arguments.destination.resolve()
    if destination.exists():
        raise SystemExit('Choose a fresh analysis destination')
    destination.mkdir(parents=True)
    data=arguments.data.resolve()
    spatial_rows,snapshots=spatial(data,destination)
    models,summary,paired,counts=energy(data,destination)
    labour=[dict(n_points=count,coefficient_fit_readings=26*count,layout_validation_readings=62*count,total_layout_readings=88*count,illustrative_seconds_per_reading=50,illustrative_layout_hours=88*count*50/3600) for count in [8,12,16,24]]
    write(destination/'acquisition_counts.csv',labour)
    selected=lambda count,method:next(row for row in spatial_rows if row['layout']=='Depth-stratified' and row['n_points']==count and row['model']==method)
    values=dict(record_group='R16',terminal_observation_count=9984,terminal_positions=208,candidate_positions=81,spatial_fits=len(spatial_rows),observed_zone_state_background_combinations=192,observed_pass_combinations=spatial_rows[0]['observed_pass_combinations'],depth12_baseline=selected(12,'Analytic baseline'),depth12_improved=selected(12,'Improved model'),depth24_improved=selected(24,'Improved model'),metered_days=48,metered_policy_summary=summary,primary_metered_contrast=models['primary'],block_pair_sensitivity=paired,metering_counts=counts,quality_endpoint='Observed steady-state snapshot task pass',energy_endpoint='End minus start cumulative electricity register for assigned observed days',input_mode='fixed_record_analysis')
    save(destination/'paper_values.json',values)
    save(destination/'execution_report.json',dict(status='completed',data_directory_role='Fixed observations and recorded instrument settings',analysis_kind='spatial_fitting_and_register_statistics',primary_unit='One classroom; 48 allocated days, 12 four-day blocks and 9984 terminal readings'))
    from mechanism_analysis import analyse
    analyse(data,destination)
    print(json.dumps(dict(spatial_fits=len(spatial_rows),observed_snapshot_combinations=192,depth12_RMSE_lx=values['depth12_improved']['RMSE_lx'],depth24_RMSE_lx=values['depth24_improved']['RMSE_lx'],metered_delta_kWh=models['primary']['estimate_kWh'],metered_CI=[models['primary']['CI_low_kWh'],models['primary']['CI_high_kWh']]),ensure_ascii=False))


if __name__=='__main__':
    main()
