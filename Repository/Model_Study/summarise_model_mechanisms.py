"""Summarise controlled numerical mechanisms without entering the observed analysis."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np

HERE=Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True);arguments=parser.parse_args()
    destination=arguments.destination.resolve()
    if destination.exists():raise SystemExit('Choose a fresh numerical-summary destination')
    destination.mkdir(parents=True)
    source=HERE/'Study/analysis_results'
    conditions=read(source/'controlled_feasibility_family.csv');families=[]
    labels={'true_instant_lower':'Benchmark at instantaneous lower endpoint','true_same_interval':'Benchmark over the same daylight interval','prediction_no_guard':'Prediction without spatial allowance','task_metric_guard':'Task-metric protection','uniform_margin':'Uniform-margin comparator'}
    for family in ['true_instant_lower','true_same_interval','prediction_no_guard','task_metric_guard','uniform_margin']:
        group=[row for row in conditions if row['family']==family]
        assert len(group)==1620
        families.append(dict(family=family,description=labels[family],total_conditions=len(group),feasible_conditions=sum(int(row['feasible_states'])>0 for row in group),evidence_role='Numerical controlled comparison; not observed classroom prevalence'))
    screen=read(source/'commissioning_decision_screen.csv');assert len(screen)==36
    summary=dict(parameter_conditions=36,common_input_days=8,policy='PU',point_count_comparison='24 minus 12 depth-stratified points',mean_energy_difference_kWh=float(np.mean([float(row['energy_difference_kWh']) for row in screen])),median_energy_difference_kWh=float(np.median([float(row['energy_difference_kWh']) for row in screen])),mean_Jauto_difference_pp=float(np.mean([float(row['Jauto_difference_pp']) for row in screen])),median_Jauto_difference_pp=float(np.median([float(row['Jauto_difference_pp']) for row in screen])),mean_active_changed_pct=float(np.mean([float(row['active_changed_pct']) for row in screen])),median_active_changed_pct=float(np.median([float(row['active_changed_pct']) for row in screen])),conditions_below_screening_scales=sum(row['decision']=='no_increment_above_declared_scales' for row in screen),families=families,scope='Separate controlled numerical mechanisms; no replacement of observed P-policy electricity or steady snapshot endpoints')
    potential=read(HERE/'Study/metering_data/field_potential_outcomes.csv');paired=[]
    for profile in range(1,49):
        before=next(row for row in potential if row['policy']=='P12' and int(row['day'])==profile)
        after=next(row for row in potential if row['policy']=='P24' and int(row['day'])==profile)
        assert before['input_seed']==after['input_seed']
        paired.append(dict(profile_id=profile,input_identifier=before['input_seed'],P12_trace_sha256=before['state_trace_sha256'],P24_trace_sha256=after['state_trace_sha256'],identical_trace=int(before['state_trace_sha256']==after['state_trace_sha256']),energy_difference_kWh=float(after['total_incremental_kWh'])-float(before['total_incremental_kWh']),evidence_role='Common-input numerical P-policy replay; not observed paired-day execution'))
    summary['P_replay_profiles']=len(paired);summary['P_identical_replay_traces']=sum(row['identical_trace'] for row in paired);summary['P_max_absolute_replay_energy_difference_kWh']=max(abs(row['energy_difference_kWh']) for row in paired)
    write(destination/'controlled_mechanism_summary.csv',families)
    write(destination/'numerical_point_count_summary.csv',[{key:value for key,value in summary.items() if key!='families'}])
    write(destination/'paired_P_replay_summary.csv',paired)
    (destination/'mechanism_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
