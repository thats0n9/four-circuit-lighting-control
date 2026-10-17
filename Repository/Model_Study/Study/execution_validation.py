"""Numerical model and evaluation routines."""
from pathlib import Path
import csv,json,sys,tempfile,shutil
from metering_statistics import process as process_metering_records,write

def process(src):
 src=Path(src);out=src/'processed';configpath=src/'execution_settings.json'
 config=json.loads(configpath.read_text('utf8')) if configpath.exists() else {}
 source=config.get('state_series_kind','unknown')
 admitted=source in ['node_local_output_readback','independent_lamp_feedback','applied_output_trace']
 expected_event_origin='lamp_feedback_update' if source=='independent_lamp_feedback' else 'node_output_update'
 execution_ok=bool(admitted and config.get('event_timestamp_definition')==expected_event_origin and config.get('frame_timestamp_definition')=='frame_acquisition' and config.get('clock_domain') and config.get('clock_alignment_verified') is True and config.get('steady_state16_verified') is True and (source!='independent_lamp_feedback' or config.get('independent_continuous_lamp_feedback') is True))
 with tempfile.TemporaryDirectory(prefix='execution_validation_') as temp:
  tmp=Path(temp)
  for n in ['calibration_segments.csv','assigned_days.csv','meter_settings.json']:shutil.copy2(src/n,tmp/n)
  for new,old in [('node_execution_events.csv','physical_output_events.csv'),('node_execution_heartbeats.csv','physical_output_heartbeats.csv')]:
   p=src/new
   if not p.exists():raise FileNotFoundError('Missing node execution interface: '+str(p))
   shutil.copy2(p,tmp/old)
  rows,models,counts=process_metering_records(tmp)
  out.mkdir(exist_ok=True)
  for p in (tmp/'processed').glob('*'):shutil.copy2(p,out/p.name)
 for r in rows:
  r['node_state_series_kind']=source;r['execution_log_valid']=int(execution_ok)
  if not execution_ok:
   r['replay_eligible']=0;r['reason']+='; node execution provenance or clock alignment not verified'
 write(out/'daily_eligibility.csv',[{k:v for k,v in r.items() if k!='state_hours'} for r in rows])
 for c in counts:c['replay_eligible']=sum(r['replay_eligible'] for r in rows if r['policy']==c['policy'])
 write(out/'policy_inclusion_counts.csv',counts)
 interpretation=dict(state_series_kind=source,event_timestamp_definition=config.get('event_timestamp_definition','unknown'),frame_timestamp_definition=config.get('frame_timestamp_definition','unknown'),clock_domain=config.get('clock_domain','unknown'),clock_alignment_verified=config.get('clock_alignment_verified') is True,steady_state16_verified=config.get('steady_state16_verified') is True,execution_log_valid=execution_ok,physical_lamp_state_continuously_verified=source=='independent_lamp_feedback' and execution_ok,replay_scope='replay unavailable' if not execution_ok else 'conditional on normal actuators and steady-state16 verification' if source=='node_local_output_readback' else 'ideal-actuator condition' if source=='applied_output_trace' else 'independent feedback',actual_energy_primary_preserved=True)
 (out/'execution_validity.json').write_text(json.dumps(interpretation,ensure_ascii=False,indent=2),encoding='utf8')
 # Return diagnostics consistent with any additional replay exclusion.
 with (out/'model_diagnostics.csv').open(encoding='utf-8-sig') as f:diag=list(csv.DictReader(f))
 elig={r['day']:r['replay_eligible'] for r in rows}
 for r in diag:r['replay_eligible']=elig[int(r['day'])]
 write(out/'model_diagnostics.csv',diag)
 return rows,models,counts

if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('Usage: python execution_validation.py input_directory')
 rows,models,counts=process(sys.argv[1]);print(json.dumps({'counts':counts,'primary_estimate_kWh':models['primary'].get('estimate_kWh'),'provenance':str(Path(sys.argv[1])/'processed/execution_validity.json')},ensure_ascii=False))
