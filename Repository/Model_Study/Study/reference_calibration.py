"""Apply the fitted reference calibration to paired policy replays."""
import numpy as np
import application_study as m
import commissioning_methods as p


def inverse(g,x,y):return np.interp(g,x,y)


def bins_table(inp,ref,edges,kind='corrected',drift=1.,lag=False):
 lamp=np.array(ref['true_reference_lamp_lx']);lf=np.array(ref['lamp_camera_equivalent_lx']);D=inp['window'];noise=inp['refgray']-m.gfunc(D)
 gray=np.clip(p.camera_forward(D[:,None]+ref['LED_response_ratio']*drift*lamp[None])+noise[:,None],0,255)
 est=inverse(gray,np.array(ref['gray_knots']),np.array(ref['lux_knots']))
 invalid=(gray<ref['gray_knots'][0]-2)|(gray>ref['gray_knots'][-1]+2)|~np.isfinite(gray)
 if kind=='raw':corrected=est
 elif lag:corrected=est[:,:,None]-lf[None,None,:]
 else:corrected=est-lf[None]
 value=corrected-ref['margin_lx'];bins=np.maximum(np.searchsorted(edges,value,side='right')-1,0).astype(np.int16)
 if lag:bins[np.broadcast_to(invalid[:,:,None],bins.shape)]=0
 else:bins[invalid]=0
 return bins

