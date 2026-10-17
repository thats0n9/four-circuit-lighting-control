"""Action differences require the effective camera-and-input health flag."""
import json,numpy as np
import analysis_core as c
from analysis_engine import fast_engine_trace
from main_analysis import summarize

def records(name):
    result=[]
    for r in c.read(c.OUT/name):
        x={}
        for k,v in r.items():
            try:x[k]=int(v)
            except ValueError:
                try:x[k]=float(v)
                except ValueError:x[k]=v
        result.append(x)
    return result

def main():
    contrasts=records('current_contrasts_daily.csv');index={(r['case'],r['layout'],r['n_points'],r['policy'],r['day']):r for r in contrasts}
    audits=[];inputs={d:dict(np.load(c.OUT/f'input_{20261339+d}.npz')) for d in range(1,9)}
    for cid in [f'R{i:02}' for i in range(1,37)]:
        ref=json.loads((c.s.R/(cid+'_reference_calibration.json')).read_text('utf-8'));look=np.load(c.OUT/(cid+'_lookups.npz'))['depth_12_PU']
        for day,inp in inputs.items():
            table=c.s.r.bins_table(inp,ref,c.EDGES[3]);ids,_,_,tele=fast_engine_trace(inp['det'],inp['valid'],table,look,inp['manuals'])
            auto=inp['work'].astype(bool).copy()
            for t,z,value,duration in inp['manuals']:
                if value==0:auto[t//5:min(len(auto),(t+duration)//5),z]=False
            requested=np.repeat(auto.any(1),5);free=(tele[:,1]==0)&(tele[:,2]==0)
            legacy=requested&inp['valid'].astype(bool)&free
            eligible=requested&tele[:,4].astype(bool)&free;den=int(eligible.sum())
            assert np.all(~eligible|legacy)
            trace=np.load(c.OUT/(cid+f'_D{day:02}_all_policy_traces.npz'))
            assert np.array_equal(ids,trace['depth_12_PU'])
            for layout in c.s.LAYOUTS:
                for n in c.s.COUNTS:
                    for policy in ['P','PU','DZU']:
                        output=trace[f'{layout}_{n}_{policy}'];changed=int(((output!=ids)&eligible).sum())
                        r=index[cid,layout,n,policy,day];r['active_controllable_frames']=den;r['changed_active_controllable_frames']=changed;r['active_changed_pct']=100*changed/max(den,1)
            audits.append(dict(case=cid,day=day,communication_valid_free_frames=int(legacy.sum()),effective_healthy_free_frames=den,
                camera_invalid_frames_removed=int((legacy&~eligible).sum()),reference_policy='depth12 PU',
                primary_quality_and_energy_unchanged=True))
        print('effective action denominator',cid,flush=True)
    c.write('current_contrasts_daily.csv',contrasts);c.write('action_health_denominator_audit.csv',audits)
    filenames=['current_control_daily.csv','current_contrasts_daily.csv','protection_terminal_coverage.csv','protection_parameters.csv','current_feasibility_scan.csv','feasibility_runtime_layers.csv','S8_current_reference_daily.csv','protection_metric_terminal_coverage.csv','protection_metric_parameters.csv','spatial_metrics.csv','spatial_noise_replicates.csv']
    summarize([records(name) for name in filenames])
    scene=c.read(c.OUT/'current_scene_summary.csv')
    current={r['case']:float(r['active_changed_pct']) for r in scene if (r['layout'],r['n_points'],r['policy'])==('depth','24','PU')}
    screen=c.OUT/'commissioning_decision_screen.csv'
    if screen.exists():
        rows=c.read(screen)
        for r in rows:r['active_changed_pct']=current[r['case']]
        c.write(screen.name,rows)

if __name__=='__main__':main()
