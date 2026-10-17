"""Orthogonal S8 diagnostics and conditional replay uncertainty, study."""
import json,math,copy,shutil
import numpy as np
import analysis_core as c
s=c.s

def oracle_study():
    rows=[]
    for cid in ['R04','R16','R28']:
        case=next(r for r in c.read(s.F/'room_scenarios.csv') if r['case']==cid)
        z,pool,_=c.load_case(case)
        ref=json.loads((s.R/(cid+'_reference_calibration.json')).read_text('utf8'))
        cal=s.fit(z,pool,s.selections()['depth',12],'improved')
        plans={}
        for name,H,day,zones,edges,cm in [
             ('B_prediction_no_guard',cal['pred'],cal['dayhat'],s.m.ZONES,c.EDGES[3],ref['margin_lx']),
             ('C_true81_camera3',z['true_H'],z['true_day'],s.m.ZONES,c.EDGES[3],ref['margin_lx']),
             ('D_true81_trueD3_wide',z['true_H'],z['true_day'],s.m.ZONES,c.EDGES[3],ref['margin_lx']),
             ('E_true81_trueD3',z['true_H'],z['true_day'],s.m.ZONES,c.EDGES[3],0.),
             ('F_true81_trueD1lx',z['true_H'],z['true_day'],s.m.ZONES,np.arange(1261.),0.),
             ('G_true208_trueD1lx',z['final_H'],z['final_day'],z['final_zones'],np.arange(1261.),0.)]:
            ds=c.deficit(H,day,zones,edges,cm)
            plans[name]=c.lookups(ds)[0]
        plans['A_currentPU3']=np.load(c.OUT/(cid+'_lookups.npz'))['depth_12_PU']
        np.savez_compressed(c.OUT/(cid+'_orthogonal_oracle_lookups.npz'),provenance=c.PROV,**plans)
        for day,seed in enumerate(range(20261360,20261364),1):
            inp=s.m.construct(seed);qt=c.quality_tables(inp,z);manual=np.array(inp['manuals'],np.int64).reshape(-1,4)
            tab=s.r.bins_table(inp,ref,c.EDGES[3])
            true3=np.broadcast_to(np.maximum(np.searchsorted(c.EDGES[3],inp['window'],side='right')-1,0)[:,None],(s.m.SECONDS,16)).astype(np.int16)
            truefine=np.broadcast_to(np.clip(np.floor(inp['window']),0,1260).astype(np.int16)[:,None],(s.m.SECONDS,16)).copy()
            for name,plan in plans.items():
                table=tab if name[0] in ['A','B','C'] else true3 if name[0] in ['D','E'] else truefine
                ids,vi,bb=s.fast_engine(inp['det'],inp['valid'],table,plan,manual)
                rows.append(dict(provenance=c.PROV,case=cid,day=day,input_seed=seed,diagnostic=name,
                    **c.evaluate(ids,inp,z,qt,vi)))
        print('orthogonal oracle diagnostics',cid,flush=True)
    c.write('S8_orthogonal_oracle_daily.csv',rows)
    summary=[]
    for cid in ['R04','R16','R28']:
        for name in plans:
            rr=[r for r in rows if r['case']==cid and r['diagnostic']==name]
            summary.append(dict(case=cid,diagnostic=name,energy_kWh_day=float(np.mean([r['total_incremental_kWh'] for r in rr])),
                 Jauto_pct=float(np.mean([r['Jauto_pct'] for r in rr]))))
    c.write('S8_orthogonal_oracle_summary.csv',summary)

def replay_uncertainty():
    rows=[]
    for cid in ['R04','R16','R28']:
        case=next(r for r in c.read(s.F/'room_scenarios.csv') if r['case']==cid)
        z,pool,_=c.load_case(case)
        ref=json.loads((s.R/(cid+'_reference_calibration.json')).read_text('utf8'))
        field=s.v.radiosity_field(float(case['length_m']),float(case['width_m']),float(case['ceiling_m']),
                'low' if case['wall_reflectance']=='0.35' else 'high',case['window'])
        raw=np.load(c.OUT/(cid+'_protection_observations.npz'))
        obs=dict(points=raw['points'],zones=raw['zones'],night=raw['night_readings'],day=raw['day_readings'],reference=raw['reference_readings'])
        for n in [12,24]:
            cal=s.fit(z,pool,s.selections()['depth',n],'improved');g=c.protect(field,cal,obs)
            lamp=s.m.STATES@cal['pred'].T
            possible=np.zeros((3,16,4),bool);certain=np.zeros_like(possible)
            lower_edges=c.EDGES[3];upper_edges=np.r_[lower_edges[1:]+2*ref['margin_lx'],1260.]
            for zone in range(4):
                ix=s.m.ZONES==zone
                for db,(lo,hi) in enumerate(zip(lower_edges,upper_edges)):
                    e1=lamp[:,ix]+lo*cal['dayhat'][ix]
                    e2=lamp[:,ix]+hi*cal['dayhat'][ix]
                    lower=e1-g['nplus'][:,zone,None]-lo*g['dplus'][zone]
                    upper=e2+g['nminus'][:,zone,None]+hi*g['dminus'][zone]
                    amlo=lower.mean(1);amhi=upper.mean(1);mnlo=lower.min(1);mnhi=upper.min(1)
                    ulo=mnlo/np.maximum(amhi,1);uhi=np.minimum(1,mnhi/np.maximum(amlo,1))
                    possible[db,:,zone]=(amlo<500)|(mnlo<300)|(ulo<.6)
                    certain[db,:,zone]=(amhi<500)|(mnhi<300)|(uhi<.6)
            for day,seed in enumerate(range(20261340,20261348),1):
                inp=s.m.construct(seed);qt=c.quality_tables(inp,z);auto=qt[-1]
                ids=np.load(c.OUT/(cid+'_reference_traces.npz'))[f'D{day:02}_{n}']
                ref_table=s.r.bins_table(inp,ref,c.EDGES[3]);den=5*int(auto.sum())
                actual=c.evaluate(ids,inp,z,qt)['Jauto_pct'];low=up=0
                # Executed bins are recalculated from the sampled applied state,
                # respecting the camera's 1 Hz sample/hold.
                sampled=ids[::5];bins=ref_table[np.arange(s.m.SECONDS),sampled]
                for phase in range(5):
                    state=ids[phase::5]
                    low+=int((certain[bins,state]&auto).sum());up+=int((possible[bins,state]&auto).sum())
                rows.append(dict(provenance=c.PROV,case=cid,n_points=n,day=day,conditional_lower_Jauto_pct=100*low/max(den,1),
                    conditional_upper_Jauto_pct=100*up/max(den,1),terminal_reference_Jauto_pct=actual,
                    interpretation='coverage-conditioned deterministic envelope; not a continuous measurement or statistical CI'))
    c.write('conditional_replay_quality_envelope.csv',rows)

def precision_and_decisions():
    # Keep the original zero-effect assigned-day example and its full processing.
    rows=[]
    for bound in [.15695352400408588,.034677215189873455,.01]:
        for sd in [.04,.08,.16,.32]:
            for total in [48,96,192,384]:
                arm=total//4;df=total-15
                half=s.v.tcrit(df+1)*sd*math.sqrt(2/arm)
                rows.append(dict(fixed_bound_kWh=bound,residual_SD_kWh=sd,total_days=total,days_per_configuration=arm,
                    random_halfwidth_kWh=half,total_halfwidth_kWh=half+bound,
                    can_resolve_0p05_kWh=int(half+bound<.05),
                    can_resolve_0p10_kWh=int(half+bound<.10),
                    role='conditional precision scenario; reduced bounds require independently supported calibration'))
    c.write('energy_precision_grid.csv',rows)
    design=c.read(c.OUT/'current_scene_summary.csv');cov=c.read(c.OUT/'protection_metric_terminal_coverage.csv')
    out=[]
    for cid in sorted({r['case'] for r in design}):
        a=next(r for r in design if (r['case'],r['layout'],r['n_points'],r['policy'])==(cid,'depth','12','PU'))
        b=next(r for r in design if (r['case'],r['layout'],r['n_points'],r['policy'])==(cid,'depth','24','PU'))
        dj=float(b['Jauto_pct'])-float(a['Jauto_pct']);de=float(b['total_incremental_kWh'])-float(a['total_incremental_kWh'])
        qualified=int(a['protection_qualified']) and int(b['protection_qualified'])
        if not qualified:decision='protection_not_qualified'
        elif dj<=-.2 and de<=.05:decision='quality_gain_candidate'
        elif de<=-.05 and dj<=.2:decision='energy_gain_candidate_pending_meter_precision'
        elif dj>.2:decision='quality_worsens'
        else:decision='no_increment_above_declared_scales'
        out.append(dict(case=cid,comparison='depth24_minus_depth12_PU',Jauto_difference_pp=dj,energy_difference_kWh=de,
          active_changed_pct=float(b['active_changed_pct']),both_protection_qualified=int(qualified),decision=decision,
          extra_point_readings=1056,extra_point_labour_hours=14.666666666666666))
    c.write('commissioning_decision_screen.csv',out)

def main():
    oracle_study();replay_uncertainty();precision_and_decisions()

if __name__=='__main__':main()
