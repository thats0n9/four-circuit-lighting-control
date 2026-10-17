"""Attribute reading-quality changes to freely controllable/restricted contexts."""
import json,numpy as np
import analysis_core as c
from analysis_engine import fast_engine_trace
s=c.s

def main():
    rows=[]
    for cid in ['R02','R08','R10','R15','R25','R28','R30']:
        case=next(r for r in c.read(s.F/'room_scenarios.csv') if r['case']==cid)
        z,pool,_=c.load_case(case);ref=json.loads((s.R/(cid+'_reference_calibration.json')).read_text('utf8'))
        look=np.load(c.OUT/(cid+'_lookups.npz'))['depth_12_PU'];tr=np.load(c.OUT/(cid+'_reference_traces.npz'))
        for day,seed in enumerate(range(20261340,20261348),1):
            inp=s.m.construct(seed);tab=s.r.bins_table(inp,ref,c.EDGES[3]);manual=np.array(inp['manuals'],np.int64)
            a,vi,bb,tele=fast_engine_trace(inp['det'],inp['valid'],tab,look,manual)
            assert np.array_equal(a,tr[f'D{day:02}_12']);b=tr[f'D{day:02}_24'];qt=c.quality_tables(inp,z)
            bad,unif,below,active,auto=qt;den=5*int(auto.sum());sec=np.arange(s.m.SECONDS)
            healthy=tele[:,4].astype(bool)
            masks={'freely_controllable':healthy&(tele[:,1]==0)&(tele[:,2]==0),
                   'manual_or_hold':healthy&((tele[:,1]!=0)|(tele[:,2]!=0)),
                   'input_invalid':~healthy}
            for label,mask in masks.items():
                delta=0;eligible=0
                for phase in range(5):
                    eligible_zone=auto&mask[phase::5,None]
                    ja=bad[a[phase::5],sec]|unif[a[phase::5],sec]
                    jb=bad[b[phase::5],sec]|unif[b[phase::5],sec]
                    delta+=int((jb&eligible_zone).sum())-int((ja&eligible_zone).sum())
                    eligible+=int(eligible_zone.sum())
                rows.append(dict(case=cid,day=day,context=label,Jauto_difference_contribution_pp=100*delta/max(den,1),
                    eligible_zone_frames=eligible,changed_state_frames=int(((a!=b)&mask&np.repeat(auto.any(1),5)).sum())))
        print('reading event context',cid,flush=True)
    c.write('reading_quality_event_context.csv',rows)

if __name__=='__main__':main()
