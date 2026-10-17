"""Control-planning and numerical evaluation definitions."""
import numpy as np
import application_study as m
PROV=''
EDGES={3:np.array([0.,120.,350.]),6:np.array([0.,60.,120.,235.,350.,805.]),12:np.array([0.,30.,60.,90.,120.,177.5,235.,292.5,350.,577.5,805.,1032.5])}
BITS=[8,4,2,1]

def planner(H,day,zones,edges,margin=0.,daymargin=0.,camera_margin=0.,true_reference=False):
    """Instantaneous lexicographic lamp selection, NOT a whole-day optimum."""
    upper=np.r_[edges[1:]+2*camera_margin,1260.] if not true_reference else edges+1
    lower=edges
    n=len(edges);avg=np.zeros((n,16,4));minimum=np.zeros_like(avg);u=np.zeros_like(avg)
    lamp=m.STATES@H.T
    for z in range(4):
        ix=zones==z
        M=margin+np.where(lower>0,daymargin,0)
        e=lamp[:,ix][None]+lower[:,None,None]*day[ix]
        eu=lamp[:,ix][None]+upper[:,None,None]*day[ix]
        avg[:,:,z]=e.mean(2)-M[:,None];minimum[:,:,z]=e.min(2)-M[:,None]
        u[:,:,z]=np.minimum((e.min(2)-M[:,None])/np.maximum(e.mean(2)+M[:,None],1),(eu.min(2)-M[:,None])/np.maximum(eu.mean(2)+M[:,None],1))
    deficit=np.maximum.reduce([(500-avg)/500,(300-minimum)/300,(.6-u)/.6,np.zeros_like(avg)])
    lookup=np.zeros((n,16,16,16),np.int8);bad=np.zeros_like(lookup)
    power=m.STATES@m.POW;codes=np.arange(16)
    for req in range(16):
        needed=m.STATES[req].astype(bool)
        dr=deficit[:,:,needed].max(2) if req else np.zeros((n,16))
        for on in range(16):
            for off in range(16):
                if on&off:continue
                valid=((codes&on)==on)&((codes&off)==0)
                score=np.where(valid[None],dr,np.inf)
                # feasible => zero deficit; otherwise deficit, then power, then code.
                order=np.lexsort((np.broadcast_to(codes,score.shape),np.broadcast_to(power,score.shape),score),axis=1)
                chosen=order[:,0];lookup[:,req,on,off]=chosen;bad[:,req,on,off]=score[np.arange(n),chosen]>0
    patterns=np.array([[lookup[j,bit,0,0] for bit in BITS] for j in range(n)],np.int8)
    union=np.zeros_like(lookup)
    for req in range(16):
        base=np.bitwise_or.reduce(patterns[:,m.STATES[req].astype(bool)],axis=1) if req else np.zeros(n,int)
        for on in range(16):
            for off in range(16):
                if not on&off:union[:,req,on,off]=(base|on)&~off
    return dict(lookup=lookup,bad=bad,union=union)


def evaluator(inp,ids,H,day,zones,stages=None,viol=0):
    if stages is None:day=day[None];stages=np.zeros(m.SECONDS,int)
    q=m.STATES[ids];qsec=q[::5]
    e=qsec@H.T+inp['window'][:,None]*day[stages]
    active=inp['work'].astype(bool);manual=np.zeros_like(active)
    for t,z,val,duration in inp['manuals']:
        if val==0:manual[t//5:min(m.SECONDS,(t+duration)//5),z]=1
    auto=active&~manual;bad=np.zeros_like(active);unif=np.zeros_like(active)
    for z in range(4):
        x=e[:,zones==z];bad[:,z]=(x.mean(1)<500)|(x.min(1)<300);unif[:,z]=x.min(1)/np.maximum(x.mean(1),1)<.6
    pp=active[:,zones];pointden=int(pp.sum());below=int(((e<500)&pp).sum())
    lighting=float(np.sum(q@m.POW)*.2/3600000)
    return dict(provenance=PROV,lighting_kWh=lighting,total_incremental_kWh=lighting+.252,transitions=int(np.abs(np.diff(q.astype(np.int16),axis=0)).sum()),work_zone_seconds=int(active.sum()),F_pct=float(100*(bad&active).sum()/max(1,active.sum())),Fauto_pct=float(100*(bad&auto).sum()/max(1,auto.sum())),U_pct=float(100*(unif&active).sum()/max(1,active.sum())),Uauto_pct=float(100*(unif&auto).sum()/max(1,auto.sum())),G_pct=100*below/max(1,pointden),manual_conflicts=int(viol))

