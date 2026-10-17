"""Draw the observation-led manuscript figures from archived result tables."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
COLOURS={'Depth-stratified':'#0072B2','Distributed':'#009E73','Window-clustered':'#D55E00','FZ12':'#777777','P8':'#D55E00','P12':'#0072B2','P24':'#8B5DA7'}


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def export(figure,destination,number):
    for suffix in ['eps','svg','pdf','png']:
        figure.savefig(destination/f'Figure_{number:02}.{suffix}',dpi=800,facecolor='white')
    plt.close(figure)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',type=Path,default=HERE/'results');parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    font_manager.findfont('Times New Roman',fallback_to_default=False)
    if args.destination.exists():raise SystemExit('Choose a fresh figure destination')
    args.destination.mkdir(parents=True)
    plt.rcParams.update({'font.family':'Times New Roman','font.size':9,'axes.labelsize':9.5,'axes.titlesize':10,'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'xtick.major.width':.6,'ytick.major.width':.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    destination=args.destination;results=args.results
    shutil.copy2(HERE/'data/Figure_01.tif',destination/'Figure_01.tif')
    shutil.copy2(HERE/'data/Figure_01.png',destination/'Figure_01.png')
    positions=read(results/'selected_positions.csv')
    with np.load(HERE/'data/protection.npz',allow_pickle=False) as archive:protection=archive['points'].copy()
    with np.load(HERE/'data/terminal.npz',allow_pickle=False) as archive:terminal=archive['points'].copy()
    figure,axes=plt.subplots(2,2,figsize=(6.25,4.8),layout='constrained')
    panels=[('a  Twelve fitting points',np.array([[float(row['x_m']),float(row['y_m'])] for row in positions if row['layout']=='Depth-stratified' and row['n_points']=='12']),COLOURS['P12'],20),('b  Twenty-four fitting points',np.array([[float(row['x_m']),float(row['y_m'])] for row in positions if row['layout']=='Depth-stratified' and row['n_points']=='24']),COLOURS['P24'],16),('c  Independent protection positions',protection,'#555555',8),('d  Independent terminal positions',terminal,'#555555',5)]
    for axis,(title,points,colour,size) in zip(axes.ravel(),panels):
        axis.scatter(points[:,0],points[:,1],s=size,c=colour,edgecolors='none')
        axis.plot([0,0],[0,8.2],color='#0072B2',lw=2)
        for value in [2.05,4.1,6.15]:axis.axhline(value,color='#b8b8b8',lw=.5,ls='--')
        axis.set(xlim=(-.2,11.2),ylim=(-.2,8.4),xlabel='Distance from window (m)',ylabel='Across-room distance (m)',title=title)
        axis.set_aspect('equal');axis.set_xticks([0,5,10]);axis.set_yticks([0,4,8])
    export(figure,destination,2)
    rows=read(results/'spatial_metrics.csv')
    figure,axes=plt.subplots(1,2,figsize=(6.25,3.25),layout='constrained')
    for axis,method,label in zip(axes,['Analytic baseline','Improved model'],['a  Analytic baseline','b  Gain–offset model']):
        for layout in ['Depth-stratified','Distributed','Window-clustered']:
            selected=[row for row in rows if row['model']==method and row['layout']==layout]
            axis.plot([int(row['n_points']) for row in selected],[float(row['RMSE_lx']) for row in selected],color=COLOURS[layout],marker={'Depth-stratified':'o','Distributed':'s','Window-clustered':'^'}[layout],ms=4,lw=1.2,label=layout)
        axis.set(xlabel='Fitting points',ylabel='RMSE against terminal readings (lx)',title=label)
        axis.set_xticks([8,12,16,24]);axis.grid(axis='y',color='#dddddd',lw=.5)
    figure.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=3,frameon=False)
    export(figure,destination,3)
    available=read(results/'observed_state_availability.csv')
    matrix=np.array([[int(row['passing_zones']) for row in available if int(row['background_lx'])==level] for level in [0,200,500]])
    figure,axis=plt.subplots(figsize=(6.25,2.5),layout='constrained')
    image=axis.pcolormesh(np.arange(17)-.5,np.arange(4)-.5,matrix,vmin=0,vmax=4,cmap='Blues',shading='flat',rasterized=False)
    axis.set_ylim(2.5,-.5)
    for row in range(3):
        for column in range(16):axis.text(column,row,str(matrix[row,column]),ha='center',va='center',color='white' if matrix[row,column]>=3 else 'black',fontsize=8.5)
    axis.set_xticks(range(16));axis.set_yticks(range(3),labels=['0','200','500']);axis.set(xlabel='Executed fixed-state code',ylabel='Snapshot reference level (lx)')
    colourbar=figure.colorbar(image,ax=axis,pad=.025,fraction=.04,ticks=range(5));colourbar.set_label('Zones meeting all three targets')
    export(figure,destination,4)
    classification=read(results/'snapshot_classification.csv')
    chosen=[next(row for row in classification if row['layout']=='Depth-stratified' and row['model']=='Improved model' and row['n_points']==str(count) and row['stage']==stage) for count,stage in [(12,'Unprotected'),(24,'Unprotected'),(12,'Protected'),(24,'Protected')]]
    figure,axes=plt.subplots(2,1,figsize=(6.25,4.7),layout='constrained',gridspec_kw={'height_ratios':[1.15,1]})
    true_positive=np.array([int(row['true_positive']) for row in chosen]);false_negative=np.array([int(row['false_negative']) for row in chosen]);false_positive=np.array([int(row['false_positive']) for row in chosen]);positions=np.arange(4)
    axes[0].bar(positions,true_positive,color='#0072B2',width=.58,label='Observed pass retained')
    axes[0].bar(positions,false_negative,bottom=true_positive,color='#e2e2e2',edgecolor='#777777',hatch='///',width=.58,label='Observed pass rejected')
    axes[0].bar(positions,false_positive,bottom=true_positive+false_negative,color='#D55E00',width=.58,label='Observed failure accepted')
    for position,row in enumerate(chosen):axes[0].text(position,50,f"{row['true_positive']} / {row['false_negative']} / {row['false_positive']}",ha='center',fontsize=8)
    axes[0].axhline(46,color='#777777',ls='--',lw=.7);axes[0].set(xticks=positions,xticklabels=['12 points\nUnprotected','24 points\nUnprotected','12 points\nProtected','24 points\nProtected'],ylabel='Snapshot combinations',ylim=(0,56),title='a  Retention, rejection and false acceptance')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,1.33),ncol=3,frameon=False,fontsize=8)
    details=read(results/'snapshot_decision_details.csv')
    for name,count,field,offset,marker,colour in [('Observed',12,'observed_mean_lx',.18,'D','#333333'),('12-point prediction',12,'predicted_mean_lx',0,'o',COLOURS['P12']),('24-point prediction',24,'predicted_mean_lx',-.18,'s',COLOURS['P24'])]:
        selected=[next(row for row in details if row['layout']=='Depth-stratified' and row['model']=='Improved model' and row['n_points']==str(count) and row['background_lx']=='0' and row['state']==str(state) and row['zone']==str(zone)) for state,zone in [(11,3),(13,2)]]
        axes[1].scatter([float(row[field]) for row in selected],np.arange(2)+offset,c=colour,marker=marker,s=25,label=name)
    axes[1].axvline(500,color='#777777',ls='--',lw=.8);axes[1].set(yticks=[0,1],yticklabels=['State 11, zone 3','State 13, zone 2'],xlabel='Zone mean illuminance (lx)',ylim=(-.5,1.5),xlim=(481,508),title='b  The two 12-point false acceptances at 0 lx background')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,1.31),ncol=3,frameon=False,fontsize=8);axes[1].grid(axis='x',color='#dddddd',lw=.5)
    export(figure,destination,5)
    days=read(results/'metered_daily.csv');blocks=read(results/'metered_block_contrasts.csv');values=json.loads((results/'paper_values.json').read_text('utf-8'))
    figure=plt.figure(figsize=(6.25,5.2),layout='constrained');grid=figure.add_gridspec(2,2,height_ratios=[1,1.05]);top=figure.add_subplot(grid[0,:]);bottom=figure.add_subplot(grid[1,0]);interval=figure.add_subplot(grid[1,1])
    for policy in ['FZ12','P8','P12','P24']:
        selected=[row for row in days if row['policy']==policy]
        top.scatter([int(row['day']) for row in selected],[float(row['direct_kWh']) for row in selected],label=policy,color=COLOURS[policy],s=16,marker={'FZ12':'D','P8':'^','P12':'o','P24':'s'}[policy])
    top.set(xlabel='Allocated measurement day',ylabel='Metered electricity (kWh/day)',title='a  Daily cumulative-register differences',xlim=(0,49));top.legend(loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.23));top.grid(axis='y',lw=.5,color='#dddddd')
    differences=np.array([float(row['delta_kWh']) for row in blocks])
    bottom.scatter(range(1,13),differences,s=21,color=COLOURS['P24']);bottom.axhline(0,color='#777777',ls='--',lw=.8);bottom.set(xlabel='Four-day block',ylabel='P24 minus P12 (kWh/day)',title='b  Within-block contrasts');bottom.set_xticks([1,4,8,12]);bottom.grid(axis='y',lw=.5,color='#dddddd')
    main=values['primary_metered_contrast'];point=main['estimate_kWh'];low=main['expanded_interval_low_kWh'];high=main['expanded_interval_high_kWh']
    interval.hlines(1,low,high,color='#777777',lw=1.2);interval.plot([low,high],[1,1],'|',color='#777777',ms=9);interval.errorbar(point,1,xerr=[[point-main['CI_low_kWh']],[main['CI_high_kWh']-point]],fmt='o',color=COLOURS['P24'],capsize=4,lw=3,ms=5)
    interval.axvline(0,color='#777777',ls='--',lw=.8);interval.set(ylim=(.5,1.55),yticks=[],xlabel='P24 minus P12 (kWh/day)',title='c  Block-adjusted contrast')
    interval.text(.02,.12,'Thick: 95% statistical interval\nThin: interval plus fixed-error bound',transform=interval.transAxes,fontsize=8.2,va='bottom');interval.set_xlim(min(low,-.05)-.05,max(high,.05)+.05)
    export(figure,destination,6)
    execution=read(results/'execution_policy_summary.csv');decomposition=read(results/'execution_contrast_decomposition.csv')
    figure=plt.figure(figsize=(6.25,5.6),layout='constrained');grid=figure.add_gridspec(2,2,height_ratios=[.72,1.8]);top=figure.add_subplot(grid[0,:]);duration=figure.add_subplot(grid[1,0]);contribution=figure.add_subplot(grid[1,1],sharey=duration)
    left=np.zeros(4);palette=['#eeeeee','#b9d6e8','#71b5d6','#2b7bba','#08306b']
    for count in range(5):
        values=np.array([float(row[f'mean_{count}_circuits_on_hours'])/14*100 for row in execution])
        top.barh(range(4),values,left=left,height=.62,color=palette[count],edgecolor='white',linewidth=.4,label=str(count));left+=values
    top.set(yticks=range(4),yticklabels=[row['policy'] for row in execution],xlabel='Recorded 14 h output duration (%)',xlim=(0,100),title='a  Distribution by number of circuits on');top.invert_yaxis();top.legend(title='Circuits on',loc='upper center',bbox_to_anchor=(.5,1.7),ncol=5,frameon=False,fontsize=8,title_fontsize=8)
    delta=np.array([float(row['delta_hours'])*60 for row in decomposition]);energy=np.array([float(row['calibration_integrated_contribution_kWh']) for row in decomposition]);colours=np.where(delta>=0,COLOURS['P24'],COLOURS['P12'])
    duration.barh(range(16),delta,color=colours,height=.7);contribution.barh(range(16),energy,color=colours,height=.7)
    duration.set(yticks=range(16),yticklabels=[str(index) for index in range(16)],ylabel='Recorded state code',xlabel='P24 minus P12 (min/day)',title='b  Mean state-duration difference');duration.invert_yaxis()
    contribution.set(xlabel='Calibrated contribution (kWh/day)',title='c  State-weighted energy calculation');contribution.tick_params(axis='y',labelleft=False)
    for axis in [duration,contribution]:axis.axvline(0,color='#777777',lw=.7);axis.grid(axis='x',color='#dddddd',lw=.5);axis.set_axisbelow(True)
    export(figure,destination,7)
    (destination/'figure_report.json').write_text(json.dumps(dict(figures=7,redrawn=6,font='Times New Roman',source='Observed_Study/results',source_mode='archived_result_tables',static_candidate_diagnostics_not_operating_policies=True,trace_power_integration_not_primary_metering=True),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
