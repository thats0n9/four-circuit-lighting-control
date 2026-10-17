"""Numerical model and evaluation routines."""
from pathlib import Path
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import csv,json,hashlib,time
W=Path(__file__).resolve().parent
import numpy as np
from numba import njit
from concurrent.futures import ProcessPoolExecutor,as_completed
import application_study as m
import commissioning_methods as p
import light_field_model as v
import sensitivity_methods as d
import reference_calibration as r
O=W/'spatial_calibration';F=W/'light_field_inputs';R=W/'control_calibration'
PROV=''
COUNTS=[8,12,16,24]
LAYOUTS=['depth','area','window']

def read(f):
 with Path(f).open(encoding='utf-8-sig') as h:return list(csv.DictReader(h))
def write(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def js(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def digest(f):return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def selections():
 out={}
 for layout in LAYOUTS:
  orders=[]
  for z in range(4):
   pool=np.flatnonzero(m.ZONES==z).tolist()
   if layout=='depth':chosen=[int(p.SEL[3*z]),int(p.SEL[3*z+2]),int(p.SEL[3*z+1])]
   else:
    if layout=='window':pool=[i for i in pool if m.PTS[i,0]<=3.0]
    center=np.array([.5 if layout=='window' else 5.5,(z+.5)*2.05])
    chosen=[min(pool,key=lambda i:(float(np.sum((m.PTS[i]-center)**2)),i))]
   while len(chosen)<6:
    # Geometry-only maximin; window clustering is a deliberately adverse layout.
    remaining=[i for i in pool if i not in chosen]
    chosen.append(min(remaining,key=lambda i:(-float(np.min(np.sum(((m.PTS[i]-m.PTS[chosen])/np.array([11.,2.05]))**2,axis=1))),i)))
   orders.append(chosen)
  for n in COUNTS:out[layout,n]=np.array([i for zz in orders for i in zz[:n//4]],int)
 assert set(out['depth',12])==set(p.SEL)
 for layout in LAYOUTS:
  for a,b in zip(COUNTS,COUNTS[1:]):assert set(out[layout,a])<=set(out[layout,b])
 return out

@njit(cache=True)
def fast_engine(det,valid,table,lookup,manuals):
 # Literal P/FZ branch of frozen study engine, five-Hz actions and one-Hz reference.
 N=len(valid);out=np.zeros(N,np.int8);bins=np.zeros(N//5,np.int16)
 bits=np.array([8,4,2,1]);oc=np.zeros(4,np.int64);ac=np.zeros(4,np.int64)
 offat=np.full(4,-1,np.int64);lock=np.full(4,-1,np.int64);held=np.zeros(4,np.int64);epoch=np.zeros(4,np.int64)
 node=0;req=0;hold=0;pending=False;pq=0;pe=np.zeros(4,np.int64);last=-999999;wasvalid=True;viol=0;mi=0;db=0
 for k in range(N):
  healthy=valid[k];sec=k//5
  if not healthy and wasvalid:
   for z in range(4):epoch[z]+=1;oc[z]=0;ac[z]=0;offat[z]=-1
   pending=False;hold=15;req=0
  wasvalid=healthy
  if pending:
   for z in range(4):
    if k>=lock[z] and pe[z]==epoch[z] and healthy:node=(node&~bits[z])|(pq&bits[z])
   pending=False
  if mi<len(manuals) and manuals[mi,0]==k:
   z=manuals[mi,1];val=manuals[mi,2];duration=manuals[mi,3];b=bits[z]
   node=(node&~b)|(b if val else 0);held[z]=val;lock[z]=k+duration;epoch[z]+=1;oc[z]=0;ac[z]=0;offat[z]=-1;req&=~b;mi+=1
  for z in range(4):
   if lock[z]==k:epoch[z]+=1;hold|=bits[z];oc[z]=0;ac[z]=0;req&=~bits[z];offat[z]=-1
  if k%5==0:db=table[sec,node];bins[sec]=db
  q=node
  if not healthy:
   for z in range(4):oc[z]=0;ac[z]=0;offat[z]=-1
  else:
   for z in range(4):
    b=bits[z]
    if k<lock[z]:continue
    if det[k,z]:
     oc[z]+=1;ac[z]=0
     if oc[z]>=5:req|=b;offat[z]=-1;hold&=~b
    else:
     oc[z]=0;ac[z]+=1
     if ac[z]>=5 and offat[z]<0:offat[z]=k+300
    if offat[z]>=0 and k>=offat[z]:req&=~b;hold&=~b
   on=0;off=0
   for z in range(4):
    b=bits[z]
    if k<lock[z] or hold&b:
     if node&b:on|=b
     else:off|=b
   target=lookup[db,req&~off,on,off];q=target if k-last>=50 else node
  for z in range(4):
   b=bits[z]
   if k<lock[z]:q=(q&~b)|(b if held[z] else 0)
  if q!=node:
   pending=True;pq=q
   for z in range(4):pe[z]=epoch[z]
   last=k
  out[k]=node
  for z in range(4):
   if k<lock[z] and bool(node&bits[z])!=bool(held[z]):viol+=1
 return out,viol,bins

def fast_planner(cal,uniform=False):
 H=cal['pred'];day=cal['dayhat'];M=cal['margin'];DM=cal['day_margin'];CM=cal['camera_margin']
 lamp=m.STATES@H.T;look=np.zeros((3,16,16,16),np.int8);deficits=[]
 for db,E in enumerate([0.,120.,350.]):
  margin=M+(DM if db else 0);e=lamp+E*day;lo=e-margin;hi=e+margin
  avg=np.column_stack([lo[:,m.ZONES==z].mean(1) for z in range(4)])
  minimum=np.column_stack([lo[:,m.ZONES==z].min(1) for z in range(4)])
  deficit=np.maximum.reduce([(500-avg)/500,(300-minimum)/300,np.zeros_like(avg)])
  if uniform:
   upper=[120+2*CM,350+2*CM,1260][db];eu=lamp+upper*day
   unif=np.column_stack([np.minimum(lo[:,m.ZONES==z].min(1)/np.maximum(hi[:,m.ZONES==z].mean(1),1),(eu[:,m.ZONES==z].min(1)-margin)/np.maximum(eu[:,m.ZONES==z].mean(1)+margin,1)) for z in range(4)])
   deficit=np.maximum(deficit,np.maximum((.6-unif)/.6,0))
  deficits.append(deficit)
 deficits=np.array(deficits);codes=np.arange(16);power=m.STATES@m.POW
 for req in range(16):
  needed=m.STATES[req].astype(bool);dr=deficits[:,:,needed].max(2) if req else np.zeros((3,16))
  for on in range(16):
   for off in range(16):
    if on&off:continue
    choices=((codes&on)==on)&((codes&off)==0);score=np.where(choices[None],dr,np.inf)
    look[:,req,on,off]=np.lexsort((np.broadcast_to(codes,score.shape),np.broadcast_to(power,score.shape),score),axis=1)[:,0]
 patterns=np.array([[look[j,b,0,0] for b in [8,4,2,1]] for j in range(3)],np.int8)
 fixed=np.zeros_like(look)
 for req in range(16):
  base=int(np.bitwise_or.reduce(patterns[0,m.STATES[req].astype(bool)])) if req else 0
  for on in range(16):
   for off in range(16):
    if not on&off:fixed[:,req,on,off]=(base|on)&~off
 return look,fixed


def fit(z,pool,sel,method,night=None,day=None,ref=None):
 night=pool['night'] if night is None else night;day=pool['day'] if day is None else day;ref=pool['reference'] if ref is None else ref
 y=np.column_stack([(night[:2,b,sel]-night[:2,0,sel]).mean(0) for b in [8,4,2,1]])
 prior=z['prior'];P=z['points'];yc=P[:,1]-P[:,1].max()/2;A=np.column_stack([np.ones(81),P[:,0],yc,yc**2])
 gain=np.maximum(0,np.sum(y*prior[sel],0)/np.sum(prior[sel]**2,0))
 lamp=np.column_stack([gain,np.zeros(4)]) if method=='original' else np.array([v.nnls2(np.column_stack([prior[sel,j],np.ones(len(sel))]),y[:,j]) for j in range(4)])
 pred=prior*lamp[:,0]+lamp[:,1];fpred=pool['final_prior']*lamp[:,0]+lamp[:,1]
 mm=float(np.max(abs(night[2:,:,sel]-((m.STATES@pred[sel].T)[None])))+.03*pred.sum(1).max()+1)
 x=ref[:4].ravel();dd=day[:4,:,sel].reshape(16,len(sel));slopes=(x[:,None]*dd).sum(0)/(x@x)
 dc=np.linalg.lstsq(A[sel],np.log(slopes),rcond=None)[0] if method=='original' else v.nnls2(np.column_stack([pool['direct_day'][sel],np.ones(len(sel))]),slopes)
 fc=z['final_points'];fy=fc[:,1]-P[:,1].max()/2
 dayhat=np.exp(A@dc) if method=='original' else pool['direct_day']*dc[0]+dc[1]
 fday=np.exp(np.column_stack([np.ones(208),fc[:,0],fy,fy**2])@dc) if method=='original' else pool['final_direct_day']*dc[0]+dc[1]
 dm=float(np.quantile(abs(ref[:4,:,None]*dayhat[sel]-dd.reshape(4,4,len(sel))),.95)+2)
 design=np.column_stack([pool['direct_day'][sel],np.ones(len(sel))]) if method=='improved' else A[sel]
 scaled=design/np.maximum(np.linalg.norm(design,axis=0),1e-12)
 return dict(pred=pred,dayhat=dayhat,margin=mm,day_margin=dm,final_pred=fpred,final_dayhat=fday,lamp=lamp,daycoef=dc,day_rank=int(np.linalg.matrix_rank(scaled)),day_condition=float(np.linalg.cond(scaled)))

def spatial(z,pool,cal):
 predicted=m.STATES@cal['final_pred'].T+np.array([0,200,500])[:,None,None]*cal['final_dayhat'][None,None]
 true=m.STATES@z['final_H'].T+np.array([0,200,500])[:,None,None]*z['final_day'][None,None]
 e=predicted-pool['validation'];te=predicted-true
 feasible=unsafe=0
 for b in range(3):
  M=cal['margin']+(cal['day_margin'] if b else 0)
  for zone in range(4):
   ix=z['final_zones']==zone;ok=(predicted[b,:,ix].T.mean(1)-M>=500)&(predicted[b,:,ix].T.min(1)-M>=300)
   actual=(true[b,:,ix].T.mean(1)>=500)&(true[b,:,ix].T.min(1)>=300)
   feasible+=int(ok.sum());unsafe+=int((ok&~actual).sum())
 return dict(RMSE_lx=float(np.sqrt(np.mean(e**2))),MAE_lx=float(abs(e).mean()),bias_lx=float(e.mean()),night_RMSE_lx=float(np.sqrt(np.mean(e[0]**2))),daylight500_RMSE_lx=float(np.sqrt(np.mean((500*(cal['final_dayhat']-z['final_day']))**2))),truth_RMSE_lx=float(np.sqrt(np.mean(te**2))),night_margin_lx=cal['margin'],day_margin_lx=cal['day_margin'],feasible_zone_states=feasible,unsafe_zone_states=unsafe,unsafe_pct=100*unsafe/max(1,feasible),positive_error_max_lx=float(te.max()),holdout_readings=9984,day_fit_rank=cal['day_rank'],day_design_condition=cal['day_condition'])

def quality_tables(inp,z):
 bad=np.zeros((16,m.SECONDS,4),bool);u=np.zeros_like(bad);below=np.zeros((16,m.SECONDS,4),np.int16)
 for zone in range(4):
  ix=z['final_zones']==zone;lamps=m.STATES@z['final_H'][ix].T;day=z['final_day'][ix]
  for s in range(16):
   E=lamps[s]+inp['window'][:,None]*day;avg=E.mean(1);minimum=E.min(1)
   bad[s,:,zone]=(avg<500)|(minimum<300);u[s,:,zone]=minimum/np.maximum(avg,1)<.6;below[s,:,zone]=(E<500).sum(1)
 active=inp['work'].astype(bool);manual=np.zeros_like(active)
 for t,zone,val,duration in inp['manuals']:
  if val==0:manual[t//5:min(m.SECONDS,(t+duration)//5),zone]=True
 return bad,u,below,active,active&~manual

def evaluate_fast(ids,inp,z,qt,viol):
 bad,u,below,active,auto=qt;ix=ids[::5];sec=np.arange(m.SECONDS);b=bad[ix,sec];uu=u[ix,sec]
 q=m.STATES[ids];pointden=int(active[:,z['final_zones']].sum());cnt=int((below[ix,sec]*active).sum())
 return dict(lighting_kWh=float(np.sum(q@m.POW)*.2/3600000),total_incremental_kWh=float(np.sum(q@m.POW)*.2/3600000)+.252,transitions=int(np.abs(np.diff(q.astype(np.int16),axis=0)).sum()),Fauto_pct=float(100*(b&auto).sum()/max(1,auto.sum())),F_pct=float(100*(b&active).sum()/max(1,active.sum())),U_pct=float(100*(uu&active).sum()/max(1,active.sum())),Uauto_pct=float(100*(uu&auto).sum()/max(1,auto.sum())),G_pct=float(100*cnt/max(1,pointden)),manual_conflicts=int(viol))



