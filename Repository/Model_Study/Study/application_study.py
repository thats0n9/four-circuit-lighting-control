"""Geometry, response transformations and fixed controlled-trial input access."""
from pathlib import Path
import math
import numpy as np
from observation_inputs import trial
HERE=Path(__file__).resolve().parent
DT=.2; SECONDS=50400; N=24; F=SECONDS*5
POW=np.array([150.,153.,154.,151.])
STATES=np.array([[int(char) for char in f'{state:04b}'] for state in range(16)],dtype=np.int8)

def stats(x):
    a=np.asarray(x,float)
    return {'n':len(a),'mean':float(a.mean()),'SD':float(a.std(ddof=1)) if len(a)>1 else 0.,'min':float(a.min()),'max':float(a.max()),'median':float(np.median(a)),'P95':float(np.quantile(a,.95))}


def mask(q):return ''.join(str(int(x)) for x in q)


def code(q):return int(q[0])*8+int(q[1])*4+int(q[2])*2+int(q[3])


def geometry():
    pts=np.array([(x,y) for y in np.linspace(.5,7.7,9) for x in np.linspace(.5,10.5,9)])
    lamps=np.array([(x,y) for y in (np.arange(4)+.5)*8.2/4 for x in (np.arange(4)+.5)*11/4])
    h=2.25; dist2=((pts[:,None]-lamps[None])**2).sum(2)+h*h
    direct=5200*.9/math.pi*h*h/dist2**2
    H=np.stack([direct[:,z*4:z*4+4].sum(1) for z in range(4)],axis=1)
    H+=.12*H.mean(0)[None]
    zones=np.minimum((pts[:,1]/2.05).astype(int),3)
    daylight=np.exp(-pts[:,0]/2.5)*(.9+.1*np.cos((pts[:,1]-4.1)*math.pi/8.2))
    # Three fixed representative points per zone; map their labels by coordinates.
    selected=[]
    for z in range(4):
        ix=np.flatnonzero(zones==z); selected.extend([int(ix[0]),int(ix[len(ix)//2]),int(ix[-1])])
    return pts,H,zones,daylight,np.array(selected)


def gfunc(e):return 255*np.sqrt(np.maximum(e,0)/(np.maximum(e,0)+1200))


def invg(g):
    a=(np.clip(g,0,254)/255)**2
    return 1200*a/(1-a)


PTS,H,ZONES,DAYF,SEL=geometry()

def construct(seed):
    return trial(seed)
