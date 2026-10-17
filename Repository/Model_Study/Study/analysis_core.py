"""Independent spatial protection and coherent 0.2 s evaluation."""
from pathlib import Path
import os,sys,csv,json,hashlib
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import spatial_fitting as s
W=Path(__file__).resolve().parent
OUT=W/'analysis_results'
PROV=''
EDGES=s.d.EDGES
BITS=np.array([8,4,2,1]);CODES=np.arange(16);POWER=s.m.STATES@s.m.POW

def read(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def write(name,rows,root=OUT):
    if not rows:return
    root.mkdir(parents=True,exist_ok=True)
    with (root/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save(name,obj,root=OUT):
    root.mkdir(parents=True,exist_ok=True)
    (root/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def protection_points(case):
    # Fixed geometric positions; independent of fitting pool and terminal grid.
    L,ww=float(case['length_m']),float(case['width_m'])
    xs=np.array([.5*L/11+1e-4,.235*L,.395*L,.575*L,.745*L,10.5*L/11-1e-4])
    pts=[]
    for zone in range(4):
        lo=max(.5*ww/8.2,zone*ww/4)+1e-4
        hi=min(7.7*ww/8.2,(zone+1)*ww/4)-1e-4
        for y in [lo,(lo+hi)/2,hi]:
            for x in xs:pts.append((x,y))
    pts=np.delete(np.asarray(pts),[22,40,58],axis=0)
    assert len(pts)==69
    return pts,np.minimum((pts[:,1]/(ww/4)).astype(int),3)

def terminal_points(case):
    L,ww=float(case['length_m']),float(case['width_m'])
    xs=.5*L/11+(np.arange(13)+.5)*(10*L/11)/13
    pts=[]
    for zone in range(4):
        lo=max(.5*ww/8.2,zone*ww/4);hi=min(7.7*ww/8.2,(zone+1)*ww/4)
        ys=lo+(np.arange(4)+.5)*(hi-lo)/4
        for y in ys:
            for x in xs:pts.append((x,y))
    pts=np.asarray(pts);assert len(pts)==208
    return pts,np.minimum((pts[:,1]/(ww/4)).astype(int),3)

def load_case(case,field=None):
    from observation_inputs import arrays
    cid=case['case'];z=dict(np.load(s.F/(cid+'_field.npz')));pool=dict(np.load(s.O/(cid+'_pool.npz')))
    if field is None:
        field=s.v.radiosity_field(float(case['length_m']),float(case['width_m']),float(case['ceiling_m']),
                 'low' if case['wall_reflectance']=='0.35' else 'high',case['window'])
    data=arrays(cid+'_final_validation.npz')
    pts,zones=terminal_points(case);hp,dp=protection_points(case)
    assert np.array_equal(pts,data['points']) and np.array_equal(zones,data['zones'])
    assert np.min(np.linalg.norm(pts[:,None]-z['points'][None],axis=2))>1e-6
    assert np.min(np.linalg.norm(pts[:,None]-hp[None],axis=2))>1e-6
    assert data['validation_readings'].shape==(3,16,208)
    assert np.isfinite(data['validation_readings']).all()
    z.update(final_points=data['points'],final_zones=data['zones'],final_H=data['true_H'],final_day=data['true_day'])
    pool['final_prior']=data['prior'];pool['final_direct_day']=data['direct_day']
    pool['validation']=data['validation_readings']
    return z,pool,field

def predict_at(field,cal,points,method='improved'):
    _,_,prior,direct=s.v.field_at(field,points)
    H=prior*cal['lamp'][:,0]+cal['lamp'][:,1]
    if method=='improved':day=direct*cal['daycoef'][0]+cal['daycoef'][1]
    else:
        center=field['points'][:,1].max()/2
        y=points[:,1]-center
        day=np.exp(np.column_stack([np.ones(len(points)),points[:,0],y,y*y])@cal['daycoef'])
    return H,day

def make_protection(case,field,z):
    from observation_inputs import arrays
    data=arrays(case['case']+'_protection_observations.npz')
    pts,zones=protection_points(case)
    assert np.array_equal(pts,data['points']) and np.array_equal(zones,data['zones'])
    assert np.min(np.linalg.norm(pts[:,None]-z['points'][None],axis=2))>1e-6
    assert np.min(np.linalg.norm(pts[:,None]-z['final_points'][None],axis=2))>1e-6
    assert data['night_readings'].shape==(2,16,69) and data['day_readings'].shape==(2,4,69)
    np.savez_compressed(OUT/(case['case']+'_protection_observations.npz'),**data,provenance=PROV)
    return dict(points=pts,zones=zones,night=data['night_readings'],day=data['day_readings'],
                reference=data['reference_readings'],true_H=data['true_H'],true_day=data['true_day'])

def protect(field,cal,obs):
    hp,dp=predict_at(field,cal,obs['points']);lamp=s.m.STATES@hp.T
    mean_night=obs['night'].mean(0)
    x=obs['reference'].ravel();y=obs['day'].reshape(8,69)
    slope=(x[:,None]*y).sum(0)/(x@x)
    # Two-sided pointwise envelopes are labelled by the physical component.
    # The daylight coefficient bound includes uncertainty in numerator/reference.
    plus=np.zeros((16,4));minus=np.zeros_like(plus);dplus=np.zeros(4);dminus=np.zeros(4)
    for zone in range(4):
        ix=obs['zones']==zone
        night_u=.03*mean_night[:,ix]+1.
        plus[:,zone]=np.maximum(0,(lamp[:,ix]-mean_night[:,ix]+night_u).max(1))
        minus[:,zone]=np.maximum(0,(mean_night[:,ix]-lamp[:,ix]+night_u).max(1))
        du=.062*slope[ix]+.01
        dplus[zone]=max(0,float((dp[ix]-slope[ix]+du).max()))
        dminus[zone]=max(0,float((slope[ix]-dp[ix]+du).max()))
    return dict(nplus=plus,nminus=minus,dplus=dplus,dminus=dminus,
                observed_night=mean_night,observed_day=slope,protection_zones=obs['zones'])

def metric_guard(cal,obs,edges,cm):
    """One-sided TASK METRIC envelopes; shared meter gain cancels in U.

    Independent protection positions estimate spatial task statistics. Endpoint
    and a fixed 10 lx daylight sweep calibrate a bin-wise maximum; held-out
    terminal metrics verify that spatial and discretization errors are covered.
    These are empirical acceptance envelopes, not a distribution-free guarantee.
    """
    guards=np.zeros((len(edges),16,4,3))
    means=obs['night'].mean(0)
    x=obs['reference'].ravel();y=obs['day'].reshape(8,69)
    slope=(x[:,None]*y).sum(0)/(x@x)
    lamps=s.m.STATES@cal['pred'].T
    highs=np.r_[np.asarray(edges)[1:]+2*cm,1260.]
    for zone in range(4):
        ix=s.m.ZONES==zone;vx=obs['zones']==zone
        for db,(lo,hi) in enumerate(zip(edges,highs)):
            levels=np.unique(np.r_[lo,np.arange(np.ceil(lo/10)*10,hi,10),hi])
            pred=lamps[:,ix][None]+levels[:,None,None]*cal['dayhat'][ix]
            observed=means[:,vx][None]+levels[:,None,None]*slope[vx]
            pm=pred.mean(2);pn=pred.min(2);pu=pn/np.maximum(pm,1)
            om=observed.mean(2);on=observed.min(2);ou=on/np.maximum(om,1)
            # 3% common meter gain and 1 lx floor affect maintained illuminance.
            # U is scale-invariant; 0.01 covers repeat/minimum and spatial sampling.
            guards[db,:,zone,0]=np.maximum(0,(pm-om+.03*om+1).max(0))
            guards[db,:,zone,1]=np.maximum(0,(pn-on+.03*on+1).max(0))
            guards[db,:,zone,2]=np.maximum(0,(pu-ou+.01).max(0))
    return guards

def deficit(H,day,zones,edges,cm=0.,guard=None,uniform=True):
    lamp=s.m.STATES@H.T
    low=np.asarray(edges);high=np.r_[low[1:]+2*cm,1260.]
    mean=np.zeros((len(low),16,4));minimum=np.zeros_like(mean);unif=np.zeros_like(mean)
    for zone in range(4):
        ix=zones==zone
        for db,(lo,hi) in enumerate(zip(low,high)):
            a=lamp[:,ix]+lo*day[ix];b=lamp[:,ix]+hi*day[ix]
            np_=0 if guard is None else guard['nplus'][:,zone]
            nm_=0 if guard is None else guard['nminus'][:,zone]
            dp=0 if guard is None else guard['dplus'][zone]
            dm=0 if guard is None else guard['dminus'][zone]
            mean[db,:,zone]=a.mean(1)-np_-lo*dp
            minimum[db,:,zone]=a.min(1)-np_-lo*dp
            # min/mean is quasiconcave in scalar daylight; endpoints suffice
            # for affine fields with positive denominator, not arbitrary weather.
            u1=(a.min(1)-np_-lo*dp)/np.maximum(a.mean(1)+nm_+lo*dm,1)
            u2=(b.min(1)-np_-hi*dp)/np.maximum(b.mean(1)+nm_+hi*dm,1)
            unif[db,:,zone]=np.minimum(u1,u2)
            if guard is not None and 'metric' in guard:
                mean[db,:,zone]=a.mean(1)-guard['metric'][db,:,zone,0]
                minimum[db,:,zone]=a.min(1)-guard['metric'][db,:,zone,1]
                unif[db,:,zone]=np.minimum(a.min(1)/np.maximum(a.mean(1),1),b.min(1)/np.maximum(b.mean(1),1))-guard['metric'][db,:,zone,2]
    ds=np.maximum.reduce([(500-mean)/500,(300-minimum)/300,np.zeros_like(mean)])
    if uniform:ds=np.maximum(ds,np.maximum((.6-unif)/.6,0))
    return ds

def lookups(ds):
    n=ds.shape[0];joint=np.zeros((n,16,16,16),np.int8);dz=np.zeros_like(joint)
    feasible=np.zeros_like(joint,dtype=bool);unrestricted=np.zeros((n,16),bool)
    requests=np.zeros((n,16,16))
    for req in range(16):
        requested=s.m.STATES[req].astype(bool)
        requests[:,req]=ds[:,:,requested].max(2) if req else 0
        unrestricted[:,req]=(requests[:,req]<=1e-12).any(1)
    # Candidate ordering is fixed lexicographically: quality, power, code.
    for req in range(16):
        for on in range(16):
            for off in range(16):
                if on&off:continue
                valid=((CODES&on)==on)&((CODES&off)==0)
                score=np.where(valid[None],requests[:,req],np.inf)
                order=np.lexsort((np.broadcast_to(CODES,score.shape),np.broadcast_to(POWER,score.shape),score),axis=1)
                joint[:,req,on,off]=order[:,0]
                feasible[:,req,on,off]=(score<=1e-12).any(1)
    singles=np.array([[joint[j,b,0,0] for b in BITS] for j in range(n)],np.int8)
    for req in range(16):
        wanted=s.m.STATES[req].astype(bool)
        base=np.bitwise_or.reduce(singles[:,wanted],axis=1) if req else np.zeros(n,np.int8)
        for on in range(16):
            for off in range(16):
                if on&off:continue
                union=(base|on)&~off
                good=requests[np.arange(n),req,union]<=1e-12
                # Repair globally, adding circuits first; otherwise full joint repair.
                valid=((CODES[None]&union[:,None])==union[:,None])&((CODES&on)==on)&((CODES&off)==0)
                score=np.where(valid,requests[:,req],np.inf)
                ordering=np.lexsort((np.broadcast_to(CODES,score.shape),np.broadcast_to(POWER,score.shape),score),axis=1)
                repair=ordering[:,0]
                add_good=score[np.arange(n),repair]<=1e-12
                dz[:,req,on,off]=np.where(good,union,np.where(add_good,repair,joint[:,req,on,off]))
    return joint,dz,feasible,unrestricted

def coverage(z,cal,g,layout,n,case):
    h=s.m.STATES@cal['final_pred'].T;true=s.m.STATES@z['final_H'].T
    rows=[]
    for zone in range(4):
        ix=z['final_zones']==zone
        for component,level in [('night',0.),('daylight200',200.),('daylight500',500.),('combined1260',1260.)]:
            if component.startswith('daylight'):
                err=level*(cal['final_dayhat'][ix]-z['final_day'][ix])[None]
                lower=level*g['dplus'][zone];upper=level*g['dminus'][zone]
            else:
                err=h[:,ix]-true[:,ix]+level*(cal['final_dayhat'][ix]-z['final_day'][ix])[None]
                lower=g['nplus'][:,zone,None]+level*g['dplus'][zone]
                upper=g['nminus'][:,zone,None]+level*g['dminus'][zone]
            rows.append(dict(provenance=PROV,case=case,layout=layout,n_points=n,zone=zone+1,
                    component=component,values=err.size,positive_exceed_pct=float(100*np.mean(err>lower+1e-9)),
                    negative_exceed_pct=float(100*np.mean(-err>upper+1e-9)),
                    two_sided_coverage_pct=float(100*np.mean((err<=lower+1e-9)&(-err<=upper+1e-9))),
                    positive_error_max_lx=float(err.max())))
    return rows

def metric_coverage(z,cal,g,edges,cm,layout,n,case):
    rows=[];highs=np.r_[np.asarray(edges)[1:]+2*cm,1260.]
    lamp=s.m.STATES@cal['pred'].T;truth=s.m.STATES@z['final_H'].T
    for db,(lo,hi) in enumerate(zip(edges,highs)):
        levels=np.unique(np.r_[lo,np.arange(np.ceil(lo/25)*25,hi,25),hi])
        for zone in range(4):
            ix=s.m.ZONES==zone;tx=z['final_zones']==zone
            pp=lamp[:,ix][None]+levels[:,None,None]*cal['dayhat'][ix]
            tt=truth[:,tx][None]+levels[:,None,None]*z['final_day'][tx]
            pred=np.stack([pp.mean(2),pp.min(2),pp.min(2)/np.maximum(pp.mean(2),1)],axis=-1)
            actual=np.stack([tt.mean(2),tt.min(2),tt.min(2)/np.maximum(tt.mean(2),1)],axis=-1)
            lower=pred-g['metric'][db,:,zone][None]
            feasible=(lower>=np.array([500.,300.,.6])).all(2)
            real=(actual>=np.array([500.,300.,.6])).all(2)
            for j,name in enumerate(['mean','minimum','uniformity']):
                err=lower[:,:,j]-actual[:,:,j]
                rows.append(dict(provenance=PROV,case=case,layout=layout,n_points=n,zone=zone+1,
                    day_bin=db,metric=name,values=err.size,one_sided_coverage_pct=float(100*np.mean(err<=1e-9)),
                    maximum_lower_bound_overstatement=float(err.max()),
                    accepted_zone_state_levels=int(feasible.sum()),false_accept_zone_state_levels=int((feasible&~real).sum())))
    return rows

def quality_tables(inp,z):
    # Terminal field, distinct from planning and protection positions.
    nsec=s.m.SECONDS
    bad=np.zeros((16,nsec,4),bool);ubad=np.zeros_like(bad);below=np.zeros((16,nsec,4),np.int16)
    for zone in range(4):
        ix=z['final_zones']==zone;lamp=s.m.STATES@z['final_H'][ix].T
        day=z['final_day'][ix]
        for state in range(16):
            e=lamp[state]+inp['window'][:,None]*day
            av=e.mean(1);mi=e.min(1)
            bad[state,:,zone]=(av<500)|(mi<300)
            ubad[state,:,zone]=mi/np.maximum(av,1)<.6
            below[state,:,zone]=(e<500).sum(1)
    active=inp['work'].astype(bool);manual=np.zeros_like(active)
    for t,zone,value,duration in inp['manuals']:
        if value==0:manual[t//5:min(nsec,(t+duration)//5),zone]=1
    return bad,ubad,below,active,active&~manual

def evaluate(ids,inp,z,qt,viol=0):
    b,u,below,active,auto=qt;nsec=s.m.SECONDS;seconds=np.arange(nsec)
    ac=int(active.sum())*5;au=int(auto.sum())*5
    badall=bauto=ua=uall=jointall=jointauto=cnt=0
    for phase in range(5):
        st=ids[phase::5];bb=b[st,seconds];uu=u[st,seconds]
        badall+=int((bb&active).sum());bauto+=int((bb&auto).sum())
        uall+=int((uu&active).sum());ua+=int((uu&auto).sum())
        jointall+=int(((bb|uu)&active).sum());jointauto+=int(((bb|uu)&auto).sum())
        cnt+=int((below[st,seconds]*active).sum())
    q=s.m.STATES[ids];lamp=float(np.sum(q@s.m.POW)*.2/3600000)
    return dict(total_incremental_kWh=lamp+.252,lighting_kWh=lamp,
        transitions=int(np.abs(np.diff(q.astype(np.int16),axis=0)).sum()),
        reading_zone_seconds=ac*.2,auto_reading_zone_seconds=au*.2,manual_reading_zone_seconds=(ac-au)*.2,
        F_pct=100*badall/max(ac,1),Fauto_pct=100*bauto/max(au,1),U_pct=100*uall/max(ac,1),
        Uauto_pct=100*ua/max(au,1),J_pct=100*jointall/max(ac,1),Jauto_pct=100*jointauto/max(au,1),
        Fmanual_pct=100*(badall-bauto)/max(ac-au,1),Jmanual_pct=100*(jointall-jointauto)/max(ac-au,1),
        Fauto_contribution_pct=100*bauto/max(ac,1),Fmanual_contribution_pct=100*(badall-bauto)/max(ac,1),
        Jauto_contribution_pct=100*jointauto/max(ac,1),Jmanual_contribution_pct=100*(jointall-jointauto)/max(ac,1),
        G_pct=100*cnt/max(5*int(active[:,z['final_zones']].sum()),1),manual_conflicts=int(viol))

def engine_source():
    txt=(W/'spatial_fitting.py').read_text('utf-8')
    txt=txt[txt.index('@njit(cache=True)'):txt.index('\ndef fast_planner')]
    txt=txt.replace('def fast_engine(', 'def fast_engine_trace(',1)
    txt=txt.replace('N=len(valid);out=', 'N=len(valid);tele=np.zeros((N,7),np.int16);out=',1)
    txt=txt.replace('  healthy=valid[k];sec=k//5','  on=0;off=0;dwell_ok=k-last>=50\n  healthy=valid[k];sec=k//5',1)
    txt=txt.replace('  out[k]=node','  out[k]=node\n  tele[k]=np.array([req,on,off,dwell_ok,healthy,db,node])',1)
    txt=txt.replace(' return out,viol,bins',' return out,viol,bins,tele',1)
    return 'import numpy as np\nfrom numba import njit\n'+txt

def describe(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()),
                improved=int((a<-1e-9).sum()),equal=int((abs(a)<=1e-9).sum()),worse=int((a>1e-9).sum()))

from camera_reference_model import qualified_bins_table
from analysis_engine import fast_engine_trace
s.R=W/'camera_reference'
s.r.bins_table=qualified_bins_table
def qualified_engine(det,valid,table,lookup,manuals):
    result=fast_engine_trace(det,valid,table,lookup,manuals)
    return result[:3]
s.fast_engine=qualified_engine
