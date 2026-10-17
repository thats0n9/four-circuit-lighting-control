"""Single numerical source for manuscript tables, figures and supplements."""
import json
from collections import Counter
import numpy as np
import analysis_core as c
s=c.s

def main():
    spatial=c.read(c.OUT/'spatial_metrics.csv');design=c.read(c.OUT/'current_design_summary.csv')
    scene=c.read(c.OUT/'current_scene_summary.csv');dec=c.read(c.OUT/'commissioning_decision_screen.csv')
    grids=c.read(c.OUT/'current_feasibility_scan.csv');layers=c.read(c.OUT/'feasibility_runtime_layers.csv')
    cov=c.read(c.OUT/'protection_terminal_coverage.csv');metric=c.read(c.OUT/'protection_metric_terminal_coverage.csv')
    optics=c.read(c.OUT/'off_model_spatial_metrics.csv');sens=c.read(c.OUT/'session_sensitivity_daily.csv')
    day=c.read(c.OUT/'S8_current_reference_summary.csv');labour=c.read(c.OUT/'commissioning_labour_budget.csv')
    out=dict(spatial=[],current=design,labour=labour,S8=day,decisions=dict(Counter(r['decision'] for r in dec)),
       cases=36,current_days=8,new_control_records=len(c.read(c.OUT/'current_control_daily.csv')),
       current_spatial_fits=len(spatial),noise_fit_records=len(c.read(c.OUT/'spatial_noise_replicates.csv')),
       component_coverage_min_pct=min(float(r['two_sided_coverage_pct']) for r in cov),
       metric_full_range_coverage_min_pct=min(float(r['one_sided_coverage_pct']) for r in metric),
       S8_records=len(c.read(c.OUT/'S8_current_reference_daily.csv')),
       strict_depth12_feasible=sum(int(r['feasible_states'])>0 for r in grids if (r['layout'],r['n_points'],r['policy'])==('depth','12','PU')),
       strict_depth12_total=1620,layer_summary={})
    for layout in s.LAYOUTS:
        for n in s.COUNTS:
            for method in ['original','improved']:
                rr=[r for r in spatial if (r['layout'],r['n_points'],r['method'])==(layout,str(n),method)]
                rec=dict(layout=layout,n_points=n,method=method)
                for k in ['RMSE_lx','truth_RMSE_lx','night_RMSE_lx','daylight500_RMSE_lx','night_margin_lx','day_margin_lx']:
                    rec[k]=float(np.mean([float(r[k]) for r in rr]))
                out['spatial'].append(rec)
    for k in ['automatic_request_frames','guarded_infeasible_frames','physical_frames','day_interval_frames','model_frames','protection_frames','manual_or_hold_frames','dwell_blocked_feasible_frames']:
        out['layer_summary'][k]=sum(int(r[k]) for r in layers)
    out['layer_summary']['sampled_case_days']=len(layers)
    out['delta24_12']={}
    for k in ['Jauto_difference_pp','energy_difference_kWh','active_changed_pct']:
        out['delta24_12'][k]=c.describe([float(r[k]) for r in dec])
    # Current PU and day-aware fixed-zone baseline are always matched in model,
    # geometry, input days, protection and output guards.
    contrasts=[]
    for cid in sorted({r['case'] for r in scene}):
        pu=next(r for r in scene if (r['case'],r['layout'],r['n_points'],r['policy'])==(cid,'depth','12','PU'))
        dz=next(r for r in scene if (r['case'],r['layout'],r['n_points'],r['policy'])==(cid,'depth','12','DZU'))
        contrasts.append(dict(case=cid,DZU_minus_PU_energy_kWh=float(dz['total_incremental_kWh'])-float(pu['total_incremental_kWh']),
            DZU_minus_PU_Jauto_pp=float(dz['Jauto_pct'])-float(pu['Jauto_pct'])))
    c.write('matched_competitive_baseline.csv',contrasts)
    out['competitive']={k:c.describe([r[k] for r in contrasts]) for k in ['DZU_minus_PU_energy_kWh','DZU_minus_PU_Jauto_pp']}
    out['optics']=[]
    for kind in ['angular_1p6','window_nonuniform']:
        for method in ['original','improved']:
            rr=[r for r in optics if (r['optical_condition'],r['layout'],r['n_points'],r['method'])==(kind,'depth','12',method)]
            out['optics'].append(dict(condition=kind,method=method,mean_combined_RMSE_lx=float(np.mean([float(r['combined_truth_RMSE_lx']) for r in rr])),
               false_accept=sum(int(r['false_accept_zone_state_levels']) for r in rr)))
    sdrows=[]
    for cid in ['R04','R16','R28']:
        for label in sorted({r['session_condition'] for r in sens}):
            a=[r for r in sens if (r['case'],r['session_condition'],r['n_points'],r['policy'])==(cid,label,'12','PU')]
            b=[r for r in sens if (r['case'],r['session_condition'],r['n_points'],r['policy'])==(cid,label,'24','PU')]
            sdrows.append(dict(case=cid,session_condition=label,
                energy_difference_kWh=float(np.mean([float(r['total_incremental_kWh']) for r in b])-np.mean([float(r['total_incremental_kWh']) for r in a])),
                Jauto_difference_pp=float(np.mean([float(r['Jauto_pct']) for r in b])-np.mean([float(r['Jauto_pct']) for r in a]))))
    c.write('session_sensitivity_contrasts.csv',sdrows);out['sensitivity']=sdrows
    out['R16_meter_example']=json.loads((c.W/'metering_data/metering_example/processed/actual_day_models.json').read_text('utf8'))['primary']
    out['all_false_accept']=sum(int(r['false_accept_zone_state_levels']) for r in metric if r['metric']=='mean')
    out['all_accept']=sum(int(r['accepted_zone_state_levels']) for r in metric if r['metric']=='mean')
    c.save('paper_values.json',out)
    print(json.dumps({k:out[k] for k in ['strict_depth12_feasible','component_coverage_min_pct','delta24_12','decisions','competitive','optics','all_false_accept','all_accept']},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
