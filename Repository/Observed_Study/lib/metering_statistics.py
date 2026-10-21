"""Cumulative-register statistics and recorded-output diagnostic eligibility."""
from pathlib import Path
import csv,json,math,sys
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
import numpy as np
from statistical_utils import tcrit
def read(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(p,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with Path(p).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r.get(k,'') for k in keys} for r in rows)
def optional_number(x):
 try:
  value=float(x)
  return value if math.isfinite(value) else None
 except (ValueError,TypeError):return None

def calibration_power_centres(segments,resolution):
 step=Decimal(str(resolution))
 if not step.is_finite() or step<=0:raise ValueError('Positive finite register resolution required')
 tolerance=max(Decimal('1e-12'),step*Decimal('1e-9'))
 totals=[0]*16;durations=[Fraction(0) for state in range(16)]
 for segment in segments:
  state=int(segment['state']);duration=Fraction(str(segment['duration_h']))
  if state not in range(16) or duration<=0:raise ValueError('Invalid state or calibration duration')
  endpoints=[]
  for field in ['start_register_kWh','end_register_kWh']:
   value=Decimal(str(segment[field]))
   if not value.is_finite():raise ValueError('Nonfinite calibration endpoint')
   ticks=int((value/step).to_integral_value(rounding=ROUND_HALF_EVEN))
   if abs(value-Decimal(ticks)*step)>tolerance:raise ValueError('Calibration endpoint is not on the declared register grid')
   endpoints.append(ticks)
  if endpoints[1]<endpoints[0]:raise ValueError('Decreasing calibration register')
  totals[state]+=endpoints[1]-endpoints[0];durations[state]+=duration
 if any(duration<=0 for duration in durations):raise ValueError('Every state requires calibration exposure')
 return np.array([float(Fraction(total)*Fraction(step)/duration) for total,duration in zip(totals,durations)])
def model(rows,settings,upper,adjust=False,first='P24',second='P12'):
 if first==second or first not in ['FZ12','P8','P12','P24'] or second not in ['FZ12','P8','P12','P24']:raise ValueError('Choose two distinct recorded policy labels')
 actual=[r for r in rows if r['actual_day_eligible']];eligible=[r for r in actual if not adjust or (r['occupied_zone_hours'] is not None and r['daylight_reference_lxh'] is not None)];out=dict(n=len(eligible),actual_eligible_days=len(actual),missing_covariates_excluded_only_sensitivity=len(actual)-len(eligible),contrast=first+' minus '+second,model='block + assigned strategy'+(' + reading-zone hours + daylight integral' if adjust else ''),covariate_role='sensitivity' if adjust else 'primary',interval='day-level t interval; fixed instrument envelope reported separately')
 blocks=sorted(set(r['block'] for r in eligible));remaining=[policy for policy in ['FZ12','P8','P12','P24'] if policy not in [first,second]];policies=[remaining[0],second,remaining[1],first]
 if not all(any(r['policy']==p for r in eligible) for p in policies):out['status']='contrast_not_estimable';return out,[]
 X=np.array([[1.]+[int(r['block']==j) for j in blocks[1:]]+[int(r['policy']==p) for p in policies[1:]]+([r['occupied_zone_hours'],r['daylight_reference_lxh']/1000] if adjust else []) for r in eligible]);y=np.array([r['direct_kWh'] for r in eligible]);rank=int(np.linalg.matrix_rank(X));df=len(y)-rank
 if rank!=X.shape[1] or df<=0:out.update(status='insufficient_rank_or_residual_df',rank=rank,df=df);return out,[]
 inv=np.linalg.inv(X.T@X);c=np.zeros(X.shape[1]);c[len(blocks)]=-1;c[len(blocks)+2]=1;weight=X@inv@c;beta=inv@X.T@y;estimate=float(c@beta);res=y-X@beta;hat=np.sum((X@inv)*X,axis=1);se=math.sqrt(float(res@res/df*(c@inv@c)));hc=math.sqrt(float(np.sum(weight**2*(res/np.maximum(1-hat,1e-9))**2)));critical=tcrit(df+1)
 q=settings['resolution_kWh'];gain=settings['common_gain_halfwidth'];state=settings['state_bias_halfwidth'];drift=settings['residual_drift_halfwidth'];A=gain+state+drift;maxpower=settings.get('meter_boundary_power_upper_kW',float(max(upper)));time=settings['day_endpoint_time_halfwidth_s']
 # Shared cumulative endpoint identities are collected before absolute values.
 endpoint={}
 for r,w in zip(eligible,weight):
  for key,sign in [('register_start_id',-1),('register_end_id',1)]:endpoint[r[key]]=endpoint.get(r[key],0)+sign*float(w)
 quant=.5*q*sum(abs(a) for a in endpoint.values());clock=2*time/3600*maxpower*float(abs(weight).sum())
 common=gain*abs(estimate)/(1-A)
 # Applicable without any output log: worst-case state-dependent meter bias.
 stateworst=state/(1-A)*float(np.sum(abs(weight)*y));driftbound=drift/(1-A)*float(np.sum(abs(weight)*y));envelope=quant+clock+common+stateworst+driftbound
 # A narrower shared-state bound is available only when all output logs
 # are complete. It is a sensitivity, never required for primary inclusion.
 observedstate=None
 if all(r.get('state_hours') is not None and r['log_complete'] for r in eligible):
  aggregate=np.sum(np.array([r['state_hours'] for r in eligible])*weight[:,None],axis=0);observedstate=state*float(abs(aggregate*upper).sum())
 diagnostics=[]
 for r,w,e,h,fit in zip(eligible,weight,res,hat,X@beta):diagnostics.append(dict(provenance=r.get('provenance',''),day=r['day'],block=r['block'],policy=r['policy'],direct_kWh=r['direct_kWh'],fitted_kWh=float(fit),residual_kWh=float(e),leverage=float(h),contrast_weight=float(w),actual_day_eligible=1,execution_eligible=r['execution_eligible']))
 completecov=[r for r in eligible if r['occupied_zone_hours'] is not None and r['daylight_reference_lxh'] is not None];occ=[r['occupied_zone_hours'] for r in completecov];daylight=[r['daylight_reference_lxh']/1000 for r in completecov]
 out.update(status='estimated',rank=rank,df=df,estimate_kWh=estimate,SE_kWh=se,CI_low_kWh=estimate-critical*se,CI_high_kWh=estimate+critical*se,HC3_SE_kWh=hc,HC3_CI_low_kWh=estimate-critical*hc,HC3_CI_high_kWh=estimate+critical*hc,residual_SD_kWh=math.sqrt(float(res@res/df)),coefficients=beta.tolist(),max_leverage=float(hat.max()),quantization_fixed_bound_kWh=quant,clock_fixed_bound_kWh=clock,common_gain_fixed_bound_kWh=common,state_bias_worst_fixed_bound_kWh=stateworst,residual_drift_fixed_bound_kWh=driftbound,fixed_envelope_kWh=envelope,expanded_interval_low_kWh=estimate-critical*se-envelope,expanded_interval_high_kWh=estimate+critical*se+envelope,shared_state_sensitivity_bound_kWh=observedstate,endpoint_coefficients=endpoint,reading_zone_hours_range=[min(occ),max(occ)] if occ else None,reference_klxh_range=[min(daylight),max(daylight)] if daylight else None,covariate_correlation=float(np.corrcoef(occ,daylight)[0,1]) if len(occ)>1 and np.std(occ)>0 and np.std(daylight)>0 else None)
 return out,diagnostics
def process(src):
 src=Path(src);out=src/'processed';out.mkdir(exist_ok=True);settings=json.loads((src/'meter_settings.json').read_text('utf8'));cal=read(src/'calibration_segments.csv');days=read(src/'assigned_days.csv');events=read(src/'node_execution_events.csv');health=read(src/'node_execution_heartbeats.csv');q=settings['resolution_kWh'];g=settings['common_gain_halfwidth'];s=settings['state_bias_halfwidth'];dr=settings['residual_drift_halfwidth'];t=settings['calibration_endpoint_time_halfwidth_s'];A=g+s+dr
 if len(cal)!=48 or any(float(r['duration_h'])<=0 for r in cal):raise ValueError('48 positive calibration segments required')
 for i,r in enumerate(cal):
  if i and float(r['start_register_kWh'])!=float(cal[i-1]['end_register_kWh']):raise ValueError('Noncontinuous calibration register')
 order=np.array([int(r['state']) for r in cal]);duration=np.array([float(r['duration_h']) for r in cal]);ch=np.array([sum(duration[order==c]) for c in range(16)]);phat=calibration_power_centres(cal,q);tr=6*t/(3600*ch);upper=(phat+3*q/ch)/((1-A)*(1-tr));lower=np.maximum(0,(phat-3*q/ch)/((1+A)*(1+tr)))
 if any(sum(order==c)!=3 for c in range(16)):raise ValueError('Three calibrations per state required')
 write(out/'state_power.csv',[dict(state=c,estimated_power_kW=float(phat[c]),lower_kW=float(lower[c]),upper_kW=float(upper[c])) for c in range(16)])
 if len(set(r['day'] for r in days))!=len(days):raise ValueError('Duplicate assigned days; fault illustrations belong to separate datasets')
 rows=[];gaps=[];endpoint_readings={}
 for x in days:
  day=int(x['day']);end=(optional_number(x['duration_h']) or 0)*3600;startread=optional_number(x['register_start_kWh']);endread=optional_number(x['register_end_kWh']);validread=startread is not None and endread is not None;direct=endread-startread if validread else None;boundary=x['boundary_confirmed']=='1';clock=x['clock_confirmed']=='1';durationvalid=end>0;actual=bool(validread and direct>=0 and boundary and clock and durationvalid)
  endpointids={key:x.get(key) or f'D{day}_'+suffix for key,suffix in [('register_start_id','start'),('register_end_id','end')]}
  for key,reading in [('register_start_id',startread),('register_end_id',endread)]:
   ident=endpointids[key]
   if reading is not None:
    if ident in endpoint_readings and endpoint_readings[ident]!=reading:raise ValueError('Shared endpoint identifier has inconsistent cumulative values: '+ident)
    endpoint_readings[ident]=reading
  ev=sorted([r for r in events if int(r['day'])==day],key=lambda r:float(r['event_seconds']));hb=sorted([r for r in health if int(r['day'])==day],key=lambda r:float(r['sample_seconds']));covered=0.;maxgap=0.;packet=settings.get('heartbeat_period_s',10);consistency=True
  tt=np.array([float(r['event_seconds']) for r in ev]);states=np.array([int(r['state']) for r in ev]);evseq=np.array([int(r['event_sequence']) for r in ev]);eventvalid=bool(len(tt) and tt[0]==0 and np.all(np.diff(tt)>0) and tt[-1]<end and np.all((states>=0)&(states<=15)) and evseq[0]==0 and np.all(np.diff(evseq)==1))
  for a,b in zip(hb[:-1],hb[1:]):
   ta,tb=float(a['sample_seconds']),float(b['sample_seconds']);ok=(0<=ta<tb<=end and tb-ta<=packet+2*settings['log_event_time_halfwidth_s'] and int(b['sequence'])==int(a['sequence'])+1 and a['boot_id']==b['boot_id'] and a['valid']=='1' and b['valid']=='1')
   if ok:covered+=tb-ta
   else:maxgap=max(maxgap,tb-ta);gaps.append(dict(day=day,start_s=ta,end_s=tb,reason='heartbeat gap, sequence, validity or boot mismatch'))
  if eventvalid:
   for h in hb:
    second=float(h['sample_seconds']);idx=np.searchsorted(tt,min(second,end-1e-6),side='right')-1
    if 0<=second<=end and idx>=0 and (states[idx]!=int(h['applied_state']) or evseq[idx]!=int(h['last_event_sequence'])):consistency=False
  complete=bool(hb and float(hb[0]['sample_seconds'])==0 and float(hb[-1]['sample_seconds'])==end and abs(covered-end)<1e-6 and consistency and eventvalid)
  row=dict(provenance=x.get('provenance') or x.get('case') or 'unassigned_record_group',day=day,block=int(x['block']),position=int(x['position']),policy=x['policy'],applied_policy=x.get('applied_policy',x['policy']),fault=x.get('fault','none'),duration_h=end/3600,**endpointids,direct_kWh=direct,actual_day_eligible=int(actual),log_complete=int(complete),coverage_h=covered/3600,max_gap_s=maxgap,heartbeat_state_consistent=int(consistency),declared_coverage_h=x.get('valid_log_hours',''),boundary_confirmed=int(boundary),clock_confirmed=int(clock),occupied_zone_hours=optional_number(x['occupied_zone_hours']),daylight_reference_lxh=optional_number(x['daylight_reference_lxh']),execution_eligible=0,state_hours=None)
  if eventvalid:
   hours=np.bincount(states,weights=np.diff(np.r_[tt,end])/3600,minlength=16);coeff=np.zeros(49)
   for j,c in enumerate(order):coeff[j]-=hours[c]/ch[c];coeff[j+1]+=hours[c]/ch[c]
   quant=q+q/2*float(abs(coeff).sum());drift=2*dr*float(hours@upper);timing=float(np.sum(hours*upper*tr)+2*settings['day_endpoint_time_halfwidth_s']/3600*upper.max());event=settings['log_event_time_halfwidth_s']/3600*float(np.maximum(upper[states[:-1]]-lower[states[1:]],upper[states[1:]]-lower[states[:-1]]).sum());bound=quant+drift+timing+event;integrated=float(hours@phat);res=direct-integrated if direct is not None else None;numeric=bool(res is not None and abs(res)<=bound)
   row.update(state_hours=hours.tolist(),integrated_kWh=integrated,residual_kWh=res,closure_bound_kWh=bound,numeric_pass=int(numeric),execution_eligible=int(actual and complete and numeric))
  reasons=[]
  if not actual:reasons.append('cumulative reading, duration, clock or boundary invalid')
  if not complete:reasons.append('applied-output log incomplete or inconsistent')
  if row.get('numeric_pass')==0:reasons.append('energy closure outside fixed bound')
  row['reason']='; '.join(reasons) or 'metering and recorded-output diagnostics eligible';rows.append(row)
 write(out/'daily_eligibility.csv',[{k:v for k,v in r.items() if k!='state_hours'} for r in rows]);write(out/'heartbeat_gaps.csv',gaps or [dict(day='',start_s='',end_s='',reason='none')]);counts=[]
 for policy in ['FZ12','P8','P12','P24']:
  a=[r for r in rows if r['policy']==policy];counts.append(dict(policy=policy,assigned_days=len(a),actual_day_eligible=sum(r['actual_day_eligible'] for r in a),execution_eligible=sum(r['execution_eligible'] for r in a),fault_days=sum(r['fault']!='none' for r in a),missing_cumulative=sum(r['direct_kWh'] is None for r in a),incomplete_log_days=sum(not r['log_complete'] for r in a)))
 write(out/'policy_inclusion_counts.csv',counts);models={};diagnostics=[]
 for name,adjust in [('primary',False),('covariate_sensitivity',True)]:
  mo,dd=model(rows,settings,upper,adjust);models[name]=mo;diagnostics+=[dict(model=name,**r) for r in dd]
 leave=[]
 for block in sorted(set(r['block'] for r in rows)):
  mo,_=model([r for r in rows if r['block']!=block],settings,upper);leave.append(dict(omitted_block=block,estimate_kWh=mo.get('estimate_kWh',''),n=mo['n']))
 write(out/'model_diagnostics.csv',diagnostics);write(out/'leave_one_block_out.csv',leave);(out/'actual_day_models.json').write_text(json.dumps(models,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');return rows,models,counts
if __name__=='__main__':
 if len(sys.argv)!=2:raise SystemExit('Usage: python metering_statistics.py input_directory')
 rows,models,counts=process(sys.argv[1]);print(json.dumps(dict(counts=counts,primary={k:v for k,v in models['primary'].items() if k not in ['coefficients','endpoint_coefficients']}),ensure_ascii=False))
