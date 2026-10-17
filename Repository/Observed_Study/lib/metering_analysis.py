"""Actual energy for the sparse-point field allocation, preserving cumulative data."""
from pathlib import Path
import csv,json,tempfile,shutil,sys
from execution_validation import process as eligibility
from metering_statistics import model,write
import numpy as np
POLICIES=['FZ12','P8','P12','P24']
def read(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def fit_contrast(rows,settings,upper,first,second,adjust=False):
    result,diagnostics=model(rows,settings,upper,adjust,first,second)
    return result,diagnostics
def process(src):
    src=Path(src);out=src/'processed';original=read(src/'assigned_days.csv')
    if any(r['policy'] not in POLICIES for r in original):raise ValueError('Expected FZ12/P8/P12/P24 field labels')
    cfg=json.loads((src/'meter_settings.json').read_text('utf8'))
    required=['resolution_kWh','common_gain_halfwidth','state_bias_halfwidth','residual_drift_halfwidth','calibration_endpoint_time_halfwidth_s','day_endpoint_time_halfwidth_s','log_event_time_halfwidth_s','heartbeat_period_s','meter_boundary_power_upper_kW']
    if any(not isinstance(cfg.get(k),(int,float)) for k in required):raise ValueError('Enter actual numerical meter and timing parameters before analysing the observation records')
    with tempfile.TemporaryDirectory(prefix='metering_study_') as td:
        temp=Path(td)
        for name in ['calibration_segments.csv','meter_settings.json','execution_settings.json','node_execution_events.csv','node_execution_heartbeats.csv']:shutil.copy2(src/name,temp/name)
        write(temp/'assigned_days.csv',original);rows,_,_=eligibility(temp);out.mkdir(exist_ok=True)
        for name in ['state_power.csv','heartbeat_gaps.csv','execution_validity.json']:shutil.copy2(temp/'processed'/name,out/name)
    settings=json.loads((src/'meter_settings.json').read_text('utf8'));upper=np.array([float(r['upper_kW']) for r in read(out/'state_power.csv')]);models={};diagnostics=[];leave=[]
    for name,first,second in [('primary','P24','P12'),('secondary_8vs12','P8','P12'),('secondary_control','P12','FZ12')]:
        for adjusted in [False,True]:
            label=name+('_covariate_sensitivity' if adjusted else '')
            result,dd=fit_contrast(rows,settings,upper,first,second,adjusted);models[label]=result
            diagnostics.extend(dict(analysis=label,**r) for r in dd)
        for block in sorted(set(r['block'] for r in rows)):
            result,_=fit_contrast([r for r in rows if r['block']!=block],settings,upper,first,second)
            leave.append(dict(analysis=name,omitted_block=block,n=result['n'],estimate_kWh=result.get('estimate_kWh','')))
    counts=[]
    for policy in POLICIES:
        rr=[r for r in rows if r['policy']==policy]
        counts.append(dict(policy=policy,assigned_days=len(rr),actual_day_eligible=sum(r['actual_day_eligible'] for r in rr),execution_eligible=sum(r['execution_eligible'] for r in rr)))
    write(out/'daily_eligibility.csv',[{k:v for k,v in r.items() if k!='state_hours'} for r in rows]);write(out/'policy_inclusion_counts.csv',counts);write(out/'model_diagnostics.csv',diagnostics);write(out/'leave_one_block_out.csv',leave)
    (out/'actual_day_models.json').write_text(json.dumps(models,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    return rows,models,counts
if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: metering_analysis.py input_directory')
    _,models,counts=process(sys.argv[1]);print(json.dumps(dict(counts=counts,primary=models['primary']),ensure_ascii=False))
