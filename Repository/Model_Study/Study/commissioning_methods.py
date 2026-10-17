"""Fixed spatial layout and camera response used by the control model."""
import numpy as np
import application_study as m
SEL=np.array([np.flatnonzero(m.ZONES==zone)[np.argmin(np.sum((m.PTS[m.ZONES==zone]-[position,(zone+.5)*2.05])**2,axis=1))] for zone in range(4) for position in [.5,5.5,10.5]])

def camera_forward(E):return 6+242*(np.maximum(E,0)/(np.maximum(E,0)+1000))**.61

