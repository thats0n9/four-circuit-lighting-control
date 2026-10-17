import numpy as np
from numba import njit
@njit(cache=True)
def fast_engine_trace(det,valid,table,lookup,manuals):
 # study: invalidate automatic input outside calibrated camera range; local manual remains available.
 N=len(valid);tele=np.zeros((N,7),np.int16);out=np.zeros(N,np.int8);bins=np.zeros(N//5,np.int16)
 bits=np.array([8,4,2,1]);oc=np.zeros(4,np.int64);ac=np.zeros(4,np.int64)
 offat=np.full(4,-1,np.int64);lock=np.full(4,-1,np.int64);held=np.zeros(4,np.int64);epoch=np.zeros(4,np.int64)
 node=0;req=0;hold=0;pending=False;pq=0;pe=np.zeros(4,np.int64);last=-999999;wasvalid=True;viol=0;mi=0;db=0
 for k in range(N):
  on=0;off=0;dwell_ok=k-last>=50
  sec=k//5;healthy=bool(valid[k]) and table[sec,node]>=0
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
  if k%5==0:
   db=max(0,table[sec,node]);bins[sec]=db
   healthy=healthy and table[sec,node]>=0
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
  tele[k]=np.array([req,on,off,dwell_ok,healthy,db,node])
  for z in range(4):
   if k<lock[z] and bool(node&bits[z])!=bool(held[z]):viol+=1
 return out,viol,bins,tele
