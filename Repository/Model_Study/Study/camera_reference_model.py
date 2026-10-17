"""Numerical model and evaluation routines."""
from pathlib import Path
import json,csv
import numpy as np
import spatial_fitting as s
W=Path(__file__).resolve().parent;R=W/'camera_reference'
def qualified_bins_table(inp,ref,edges,kind='corrected',drift=1.,lag=False):
    # Only the corrected, state-compensated reference is used by study.
    D=inp['window'];noise=inp['refgray']-s.m.gfunc(D)
    gray=np.clip(s.p.camera_forward(D[:,None]+1.04*drift*np.asarray(ref['true_reference_lamp_lx'])[None])+noise[:,None],0,255)
    raw=np.interp(gray,ref['gray_knots'],ref['lux_knots'])-np.asarray(ref['lamp_camera_equivalent_lx'])[None]
    invalid=(gray<ref['gray_knots'][0]-2)|(gray>ref['gray_knots'][-1])|~np.isfinite(gray)
    bins=np.maximum(np.searchsorted(edges,np.maximum(0,raw-ref['margin_lx']),side='right')-1,0).astype(np.int16)
    bins[invalid]=-1
    return bins
def build_reference():
    R.mkdir(exist_ok=True);summary=[];audit=[]
    # A fixed 5 SD grayscale design envelope, a 0.5 lx D mesh, and 1 lx
    # discretisation reserve are specified independently of the eight test days.
    D=np.arange(0,1260.00001,.5)
    for p in sorted((W/'control_calibration').glob('R*_reference_calibration.json')):
        ref=json.loads(p.read_text('utf8'));original=ref['margin_lx'];errmax=0.
        for state in range(16):
            total=D+1.04*ref['true_reference_lamp_lx'][state]
            for gray_error in [-3.5,0,3.5]:
                gray=s.p.camera_forward(total)+gray_error
                ok=(gray>=ref['gray_knots'][0]-2)&(gray<=ref['gray_knots'][-1])
                residual=np.interp(gray,ref['gray_knots'],ref['lux_knots'])-ref['lamp_camera_equivalent_lx'][state]-D
                e=residual[ok];errmax=max(errmax,float(abs(e).max()))
                audit.append(dict(case=ref['case'],state=state,gray_error=gray_error,qualified_mesh_levels=int(ok.sum()),
                    residual_min_lx=float(e.min()),residual_max_lx=float(e.max())))
        ref['previous_margin_lx']=original;ref['margin_lx']=max(original,errmax+1.)
        ref['gray_design_error_bound']=3.5;ref['D_mesh_step_lx']=.5;ref['discretization_reserve_lx']=1.
        ref['margin_role']='common two-sided finite design envelope under calibrated response, grayscale bound and declared D domain'
        ref['provenance']=''
        (R/p.name).write_text(json.dumps(ref,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        summary.append(dict(case=ref['case'],previous_margin_lx=original,common_two_sided_margin_lx=ref['margin_lx'],
            mesh_absolute_residual_max_lx=errmax,gray_error_bound=3.5,D_mesh_step_lx=.5,discretization_reserve_lx=1.))
    for name,rows in [('reference_envelope_summary.csv',summary),('reference_envelope_design_scan.csv',audit)]:
        with (R/name).open('w',encoding='utf-8-sig',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    print('reference margins',min(x['common_two_sided_margin_lx'] for x in summary),max(x['common_two_sided_margin_lx'] for x in summary),flush=True)
if __name__=='__main__':build_reference()
