"""Optical basis and numerical reference calculations."""
import math
import numpy as np
import application_study as m


def tcrit(n):
    exact={8:2.364624251,12:2.20098516,16:2.131449546,24:2.068657610}
    if n in exact:return exact[n]
    v=n-1;z=1.959963984540054
    return z+(z**3+z)/(4*v)+(5*z**5+16*z**3+3*z)/(96*v*v)+(3*z**7+19*z**5+17*z**3-15*z)/(384*v**3)


def stats(a):
    a=np.array(a,float);n=len(a);sd=float(a.std(ddof=1)) if n>1 else 0.;h=tcrit(n)*sd/math.sqrt(n) if n>1 else 0.
    return dict(n=n,mean=float(a.mean()),SD=sd,low=float(a.mean()-h),high=float(a.mean()+h),min=float(a.min()),max=float(a.max()))


def patches(L,W,C,step):
    xyz=[];normal=[];area=[];face=[]
    definitions=[(0,L,1,W,2,0,[0,0,1],'floor'),(0,L,1,W,2,C,[0,0,-1],'ceiling'),
                 (1,W,2,C,0,0,[1,0,0],'window_wall'),(1,W,2,C,0,L,[-1,0,0],'far_wall'),
                 (0,L,2,C,1,0,[0,1,0],'end0'),(0,L,2,C,1,W,[0,-1,0],'end1')]
    for a,A,c,B,f,fv,n,name in definitions:
        na=math.ceil(A/step);nb=math.ceil(B/step)
        for i in range(na):
            for j in range(nb):
                q=np.zeros(3);q[a]=(i+.5)*A/na;q[c]=(j+.5)*B/nb;q[f]=fv
                xyz.append(q);normal.append(n);area.append(A/na*B/nb);face.append(name)
    return np.array(xyz),np.array(normal,float),np.array(area),np.array(face)


def coupling(target,tn,source,sn,area):
    # Receiver-to-source differential form-factor kernel. All geometry is a
    # convex, unobstructed enclosure. Coplanar contributions are exactly zero.
    r=source[None]-target[:,None];r2=np.sum(r*r,2);safe=np.maximum(r2,1e-12)
    ci=np.maximum(np.einsum('ijk,ik->ij',r,tn),0)
    cj=np.maximum(-np.einsum('ijk,jk->ij',r,sn),0)
    return ci*cj/(math.pi*safe*safe)*area[None]


def source_samples(L,W,C,window):
    xyz=[];norm=[];weights=[]
    # 16 finite 1.2m x 0.3m panels, each integrated by 4x2 area samples.
    for y in range(4):
        for x in range(4):
            for a in range(4):
                for c in range(2):
                    xyz.append([(x+.5)*L/4+((a+.5)/4-.5)*1.2,(y+.5)*W/4+((c+.5)/2-.5)*.3,C-.05])
                    norm.append([0,0,-1]);w=np.zeros(5);w[y]=4680/8;weights.append(w)
    zlo,zhi=(1.,C-.3) if window=='full' else (1.,1.+(C-1.3)/2)
    ya,yb=.15*W,.85*W
    for a in range(16):
        for c in range(8):
            xyz.append([.001,ya+(a+.5)/16*(yb-ya),zlo+(c+.5)/8*(zhi-zlo)])
            norm.append([1,0,0]);w=np.zeros(5);w[4]=(yb-ya)*(zhi-zlo)/128;weights.append(w)
    return np.array(xyz),np.array(norm,float),np.array(weights),dict(ylo=ya,yhi=yb,zlo=zlo,zhi=zhi)


def radiosity_field(L,W,C,reflect,window,step=.8):
    xyz,norm,area,face=patches(L,W,C,step)
    F0=coupling(xyz,norm,xyz,norm,area)
    # Symmetric balancing preserves Ai*Fij=Aj*Fji and enclosure row sum 1.
    S=area[:,None]*F0;d=np.ones(len(area))
    for _ in range(400):
        rows=d*(S@d);d*=np.sqrt(area/np.maximum(rows,1e-30))
        if np.max(abs(d*(S@d)/area-1))<1e-10:break
    F=d[:,None]*S*d[None]/area[:,None]
    refl={'low':dict(floor=.18,ceiling=.65,wall=.35),'high':dict(floor=.25,ceiling=.8,wall=.55)}[reflect]
    rho=np.array([refl.get(f,refl['wall']) for f in face])
    src,sn,sw,win=source_samples(L,W,C,window)
    iswindow=(face=='window_wall')&(xyz[:,1]>=win['ylo'])&(xyz[:,1]<=win['yhi'])&(xyz[:,2]>=win['zlo'])&(xyz[:,2]<=win['zhi'])
    rho[iswindow]=0
    E0=coupling(xyz,norm,src,sn,np.ones(len(src)))@sw
    B=rho[:,None]*E0
    for iteration in range(300):
        new=rho[:,None]*(E0+F@B)
        if np.max(abs(new-B))<1e-8:B=new;break
        B=new
    pts=m.PTS*np.array([L/11,W/8.2]);xyzp=np.column_stack([pts,np.full(81,.75)]);up=np.tile([0.,0.,1.],(81,1))
    ep=coupling(xyzp,up,src,sn,np.ones(len(src)))@sw+coupling(xyzp,up,xyz,norm,area)@B
    # A fixed near-window workplane reference defines the scalar daylight input.
    ref=np.array([[.5*L/11,W/2,.75]])
    refval=float((coupling(ref,np.array([[0.,0.,1.]]),src,sn,np.ones(len(src)))@sw+coupling(ref,np.array([[0.,0.,1.]]),xyz,norm,area)@B)[0,4])
    h=C-.75;lamps=np.array([[(x+.5)*L/4,(y+.5)*W/4] for y in range(4) for x in range(4)])
    r2=np.sum((pts[:,None]-lamps[None])**2,2)+h*h
    direct=4680/math.pi*h*h/r2**2
    prior=np.stack([direct[:,z*4:z*4+4].sum(1) for z in range(4)],1);prior+=.12*prior.mean(0)
    residual=float(np.max(abs(B-rho[:,None]*(E0+F@B))))
    diag=dict(patches=len(area),step_m=step,iterations=iteration+1,row_sum_error=float(np.max(abs(F.sum(1)-1))),reciprocity_error=float(np.max(abs(area[:,None]*F-area[None]*F.T))),equation_residual=residual,min_H=float(ep[:,:4].min()),reflectance_max=float(rho.max()),daylight_reference_factor=refval)
    # Black surfaces must reproduce the independent finite-area direct integral.
    black=coupling(xyzp,up,src,sn,np.ones(len(src)))@sw
    assert np.all(ep>=black-1e-8) and diag['row_sum_error']<1e-8 and diag['reciprocity_error']<1e-8 and residual<1e-7
    direct_day=(coupling(xyzp,up,src,sn,np.ones(len(src)))@sw)[:,4]
    return dict(H=ep[:,:4],day=ep[:,4]/refval,prior=prior,points=pts,diagnostics=diag,black=black[:,:4],direct=black[:,:4],direct_day=direct_day,solver=(xyz,norm,area,B,src,sn,sw,refval),geometry=(L,W,C))


def field_at(field,points):
    xyz,norm,area,B,src,sn,sw,refval=field['solver'];up=np.tile([0.,0.,1.],(len(points),1));xyzp=np.column_stack([points,np.full(len(points),.75)])
    direct=coupling(xyzp,up,src,sn,np.ones(len(src)))@sw;actual=direct+coupling(xyzp,up,xyz,norm,area)@B
    L,W,C=field['geometry'];h=C-.75;lamps=np.array([[(x+.5)*L/4,(y+.5)*W/4] for y in range(4) for x in range(4)]);r2=np.sum((points[:,None]-lamps[None])**2,2)+h*h;di=4680/math.pi*h*h/r2**2
    prior=np.stack([di[:,j*4:(j+1)*4].sum(1) for j in range(4)],1)
    # Same circuit's uniform prior component at 81-point base grid.
    base=field['prior'];base_direct=base-.12/(1.12)*base.mean(0);prior+=.12*base_direct.mean(0)
    return actual[:,:4],actual[:,4]/refval,prior,direct[:,4]


def nnls2(A,y):
    v=np.linalg.lstsq(A,y,rcond=None)[0];candidates=[]
    if np.all(v>=0):candidates.append(v)
    for k in range(2):
        a=np.zeros(2);a[k]=max(0,float(A[:,k]@y/(A[:,k]@A[:,k])));candidates.append(a)
    candidates.append(np.zeros(2))
    return min(candidates,key=lambda v:float(np.sum((A@v-y)**2)))

