"""Redraw the eleven English figures from the supplied exact plotting tables."""
from pathlib import Path
import csv,json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle,Patch
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

HERE=Path(__file__).resolve().parent
STYLE=json.loads((HERE/'figure_style.json').read_text('utf-8'))
C=STYLE['colours'];LAYOUTS=['depth','area','window']
LAYLABEL={'depth':'Depth strata','area':'Distributed','window':'Window cluster'}
MARK={'depth':'o','area':'s','window':'^'}
COL={'depth':C['teal'],'area':C['blue'],'window':C['orange']}
AUDIT=[]

def setup(font_dir=None):
    for folder in [Path(font_dir) if font_dir else Path('C:/Windows/Fonts')]:
        if folder.is_dir():
            for p in folder.glob('*.ttf'):
                if 'times' in p.name.lower():font_manager.fontManager.addfont(str(p))
    fp=font_manager.FontProperties(family=STYLE['font_family'])
    found=font_manager.findfont(fp,fallback_to_default=False)
    if font_manager.FontProperties(fname=found).get_name()!=STYLE['font_family']:
        raise RuntimeError('Times New Roman must be installed or supplied using --font-dir.')
    plt.rcParams.update({
        'font.family':'serif','font.serif':['Times New Roman'],'font.size':STYLE['font_size_pt'],
        'axes.labelsize':9,'axes.titlesize':STYLE['title_size_pt'],'axes.titleweight':'normal',
        'axes.titlepad':6,'axes.labelpad':4,'axes.edgecolor':C['ink'],'axes.linewidth':0.6,
        'text.color':C['ink'],'axes.labelcolor':C['ink'],'xtick.color':C['ink'],'ytick.color':C['ink'],
        'xtick.labelsize':STYLE['tick_size_pt'],'ytick.labelsize':STYLE['tick_size_pt'],
        'xtick.direction':'out','ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
        'legend.fontsize':STYLE['legend_size_pt'],'legend.frameon':False,'legend.handlelength':1.9,
        'legend.handletextpad':0.55,'legend.labelspacing':0.45,'legend.columnspacing':1.4,
        'lines.linewidth':STYLE['line_width_pt'],'lines.markersize':STYLE['marker_size_pt'],
        'axes.unicode_minus':True,'svg.fonttype':'none','svg.hashsalt':'BSERT-FIGURE-STYLE-3',
        'mathtext.fontset':'custom','mathtext.rm':'Times New Roman',
        'mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold','mathtext.fallback':None,
        'ps.fonttype':42,'pdf.fonttype':42,'hatch.linewidth':0.45,
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    })
    return found

def read(name):
    with (HERE/'data'/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def vals(rows,key):return np.array([float(r[key]) for r in rows])
def subset(rows,key,value):return [r for r in rows if r[key]==value]
def canvas(height=3.7):
    fig=plt.figure(figsize=(STYLE['width_mm']/25.4,height),layout='constrained')
    fig.get_layout_engine().set(w_pad=4/72,h_pad=4/72,wspace=0.10,hspace=0.10)
    return fig
def panels(legends=True,height=3.65):
    fig=canvas(height)
    if legends:
        gs=fig.add_gridspec(2,2,height_ratios=[1,.19])
        axes=[fig.add_subplot(gs[0,i]) for i in range(2)]
        feet=[fig.add_subplot(gs[1,i]) for i in range(2)]
        for a in feet:a.axis('off');a._legend_only=True
    else:
        gs=fig.add_gridspec(1,2);axes=[fig.add_subplot(gs[0,i]) for i in range(2)];feet=[None,None]
    return fig,axes,feet
def footer(ax,plot=None,handles=None,labels=None,ncol=1):
    if plot is not None:handles,labels=plot.get_legend_handles_labels()
    if not handles:return
    lg=ax.legend(handles,labels,loc='center',ncol=ncol,borderaxespad=0)
    lg._outside_data=True
    return lg
def grid(ax,axis='both'):
    ax.grid(axis=axis,color=C['grid'],lw=.42,zorder=0);ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
def line(ax,x,y,label,col,marker='o',ls='-'):
    return ax.plot(x,y,label=label,color=col,marker=marker,ls=ls,mew=.7,zorder=3)[0]

def visible_texts(fig):
    items=[]
    for ax in fig.axes:
        items.extend(t for t in ax.texts if t.get_visible() and t.get_text())
        if ax.axison:
            items.extend(t for t in [ax.title,ax.xaxis.label,ax.yaxis.label] if t.get_visible() and t.get_text())
            for positions,labels,limits in [(ax.get_xticks(),ax.get_xticklabels(),ax.get_xlim()),
                                           (ax.get_yticks(),ax.get_yticklabels(),ax.get_ylim())]:
                lo,hi=sorted(limits)
                items.extend(t for loc,t in zip(positions,labels) if lo-1e-10<=loc<=hi+1e-10 and t.get_visible() and t.get_text())
        legend=ax.get_legend()
        if legend is not None:items.extend(legend.get_texts())
    for legend in fig.legends:items.extend(legend.get_texts())
    return list(dict.fromkeys(items))

def inspect_labels(fig,renderer,name):
    texts=visible_texts(fig);pairs=[];point_hits=[];guide_hits=[]
    boxes={t:Text.get_window_extent(t,renderer) for t in texts}
    for i,t in enumerate(texts):
        for q in texts[i+1:]:
            if boxes[t].overlaps(boxes[q]):pairs.append([t.get_text(),q.get_text()])
    for ax in fig.axes:
        if getattr(ax,'_legend_only',False):continue
        for t in ax.texts:
            if t not in boxes:continue
            box=boxes[t]
            for coll in ax.collections:
                if not hasattr(coll,'get_offsets') or not hasattr(coll,'get_sizes'):continue
                offsets=np.asarray(coll.get_offsets());sizes=coll.get_sizes()
                if offsets.ndim!=2 or offsets.shape[1]!=2 or not len(sizes):continue
                for j,p in enumerate(coll.get_offset_transform().transform(offsets)):
                    radius=np.sqrt(sizes[min(j,len(sizes)-1)])/2*fig.dpi/72
                    if box.x0-radius<p[0]<box.x1+radius and box.y0-radius<p[1]<box.y1+radius:
                        point_hits.append(t.get_text());break
            for ln in ax.lines:
                if not getattr(ln,'_audit_guide',False):continue
                p=ln.get_transform().transform(ln.get_xydata())
                if len(p)==2:
                    if abs(p[0,0]-p[1,0])<.01 and box.x0<p[0,0]<box.x1 and max(box.y0,min(p[:,1]))<min(box.y1,max(p[:,1])):guide_hits.append(t.get_text())
                    if abs(p[0,1]-p[1,1])<.01 and box.y0<p[0,1]<box.y1 and max(box.x0,min(p[:,0]))<min(box.x1,max(p[:,0])):guide_hits.append(t.get_text())
    assert not pairs,(name,'Text bounding boxes intersect',pairs)
    assert not point_hits,(name,'Annotations cover data markers',point_hits)
    assert not guide_hits,(name,'Annotations intersect declared guide lines',guide_hits)
    return {'text_pair_intersections':0,'annotation_marker_intersections':0,'annotation_guide_intersections':0}
def save(fig,name,out):
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    data_axes=[a for a in fig.axes if not getattr(a,'_legend_only',False)]
    legends=[a.get_legend() for a in fig.axes if a.get_legend() is not None]+list(fig.legends)
    overlap=[]
    for lg in legends:
        lb=lg.get_window_extent(renderer)
        for ax in data_axes:
            if lb.overlaps(ax.get_window_extent(renderer)):overlap.append([lg.get_texts()[0].get_text(),ax.get_title()])
    assert not overlap,(name,'Legend overlaps data axes',overlap)
    fonts={t.get_fontproperties().get_name() for t in fig.findobj(Text) if t.get_text()}
    assert fonts=={'Times New Roman'},(name,fonts)
    clipped=[]
    fb=fig.bbox
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text() or t.get_clip_on():continue
        b=t.get_window_extent(renderer)
        if b.x0<fb.x0-1 or b.x1>fb.x1+1 or b.y0<fb.y0-1 or b.y1>fb.y1+1:clipped.append(t.get_text())
    assert not clipped,(name,'Text outside canvas',clipped)
    label_checks=inspect_labels(fig,renderer,name)
    out.mkdir(parents=True,exist_ok=True)
    for ext in STYLE['output_formats']:
        metadata={'Creator':'','CreationDate':None,'ModDate':None} if ext=='pdf' else ({'Creator':'','Date':None} if ext=='svg' else None)
        opts={'metadata':metadata} if metadata is not None else {}
        fig.savefig(out/(name+'.'+ext),dpi=STYLE['dpi'],**opts)
    AUDIT.append({'figure':name,'font_families':sorted(fonts),'legend_data_axes_overlaps':0,'canvas_text_overflows':0,'width_mm':STYLE['width_mm'],'formats':STYLE['output_formats'],**label_checks})
    plt.close(fig)


def fig1(out):
    fig=canvas(2.85);gs=fig.add_gridspec(2,2,width_ratios=[1,1.08],height_ratios=[.09,1])
    for i,title in enumerate(['(a) Four-circuit execution node','(b) Reference and execution workflow']):
        header=fig.add_subplot(gs[0,i]);header.axis('off');header._legend_only=True
        header.text(.5,.5,title,ha='center',va='center',fontsize=9.5)
    photo=fig.add_subplot(gs[1,0]);photo.imshow(plt.imread(HERE/'original_execution_node_photo.jpeg'))
    photo.set(xlim=(310,1550),ylim=(995,220));photo.axis('off');photo.set_anchor('N')
    for label,target,position in [('1',(815,522),(445,355)),('2',(1280,630),(1455,420))]:
        photo.annotate(label,xy=target,xytext=position,ha='center',va='center',fontsize=9,
            bbox={'boxstyle':'circle,pad=.22','fc':'white','ec':C['ink'],'lw':.6},
            arrowprops={'arrowstyle':'-','lw':.75,'color':'white'})
    photo.text(.5,-.07,'1  Controller     2  Interface board',transform=photo.transAxes,
               ha='center',va='top',fontsize=8.5)
    ax=fig.add_subplot(gs[1,1]);ax.set(xlim=(0,10),ylim=(0,8));ax.axis('off')
    boxes=[(.15,5.85,4.4,1.5,'Fixed-exposure reference\nGreyscale and capture time'),
           (5.0,5.85,4.85,1.5,'Local execution record\nState and output-update time'),
           (.6,3.35,8.8,1.65,'Align times; convert the reference\nSubtract the captured-state contribution'),
           (.6,.75,8.8,1.70,'Apply reference margin; select bin and state\nCheck all 16 steady states\nIlluminance and power')]
    for x,y,w,h,t in boxes:
        ax.add_patch(Rectangle((x,y),w,h,facecolor='#F8FAFA',edgecolor=C['grey'],lw=.6))
        ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8.5,linespacing=1.25)
    arrows=[((2.35,5.83),(3.1,5.02)),((7.42,5.83),(6.9,5.02)),((5,3.33),(5,2.47))]
    for p,q in arrows:ax.annotate('',xy=q,xytext=p,arrowprops={'arrowstyle':'->','color':C['grey'],'lw':.85})
    save(fig,'Fig1_execution_node_english',out)

def fig2(out):
    rows=read('Fig2_positions.csv');fig=canvas(3.80);gs=fig.add_gridspec(2,3)
    axes=[fig.add_subplot(gs[i//3,i%3]) for i in range(5)]
    key=fig.add_subplot(gs[1,2]);key.axis('off');key._legend_only=True
    for i,l in enumerate(LAYOUTS):
        rr=subset(rows,'set',l);axes[i].scatter(vals(rr,'x_m'),vals(rr,'y_m'),s=18,marker=MARK[l],
            facecolor=COL[l],edgecolor='white',lw=.35,zorder=3)
        axes[i].set_title(f'({chr(97+i)}) {LAYLABEL[l]}: 12 points')
    rr=subset(rows,'set','protection');axes[3].scatter(vals(rr,'x_m'),vals(rr,'y_m'),s=15,marker='s',
        facecolor='white',edgecolor=C['red'],lw=.65,zorder=3)
    axes[3].set_title('(d) Protection: 69 points')
    rr=subset(rows,'set','final_validation');axes[4].scatter(vals(rr,'x_m'),vals(rr,'y_m'),s=7,
        color=C['grey'],edgecolor='none',zorder=3);axes[4].set_title('(e) Final validation: 208 points')
    for i,ax in enumerate(axes):
        ax.add_patch(Rectangle((0,0),11,8.2,facecolor='none',edgecolor=C['grey'],lw=.6))
        ax.plot([0,0],[0,8.2],color=C['blue'],lw=2.0,zorder=4)
        for y in [2.05,4.1,6.15]:ax.plot([0,11],[y,y],color=C['grey'],ls=(0,(3,3)),lw=.5,zorder=1)
        ax.set(xlim=(-.25,11.25),ylim=(-.15,8.35),xticks=[0,5,10],yticks=[0,4,8]);ax.set_aspect('equal')
        for spine in ax.spines.values():spine.set_visible(False)
        if i in [0,3]:ax.set_ylabel('Lateral position (m)')
        if i in [3,4]:ax.set_xlabel('Depth from window (m)')
    footer(key,handles=[Line2D([],[],color=C['blue'],lw=2),Line2D([],[],color=C['grey'],ls=(0,(3,3)),lw=.6)],
           labels=['Window side (x = 0)','Task-zone boundary'],ncol=1)
    key.text(.5,.18,'Reference classroom: 11.0 × 8.2 m\nFour task zones along the lateral axis',
             ha='center',va='center',transform=key.transAxes,fontsize=8.5,linespacing=1.35)
    save(fig,'Fig2_layouts_and_independent_sets',out)

def fig3(out):
    rows=read('Fig3_RMSE.csv');fig,aa,feet=panels(height=3.2)
    for m,label,col,marker,ls in [('original','Analytic baseline',C['grey'],'D','--'),('improved','Improved model',C['teal'],'o','-')]:
        rr=[r for r in rows if r['panel']=='a' and r['series']==m];line(aa[0],vals(rr,'n_points'),vals(rr,'RMSE_lx'),label,col,marker,ls)
    for l,ls in zip(LAYOUTS,['-','--','-.']):
        rr=[r for r in rows if r['panel']=='b' and r['series']==l];line(aa[1],vals(rr,'n_points'),vals(rr,'RMSE_lx'),LAYLABEL[l],COL[l],MARK[l],ls)
    for ax in aa:ax.set(xticks=[8,12,16,24],xlabel='Number of fitting points',ylabel='Combined RMSE (lx)');grid(ax,'y')
    aa[0].set_title('(a) Models at common points');aa[1].set_title('(b) Layouts at equal point counts')
    for ax,foot in zip(aa,feet):footer(foot,ax)
    save(fig,'Fig3_prediction',out)

def fig4(out):
    fig,aa,_=panels(False,3.05);rr=read('Fig4a_feasibility.csv')
    labels=['Uniform margin','Unprotected\nprediction','Benchmark interval','Instantaneous\nbenchmark','Task-metric\nprotection']
    aa[0].barh(np.arange(5),vals(rr,'count'),color=[C['grey']]*4+[C['teal']],height=.58)
    aa[0].set(yticks=np.arange(5),yticklabels=labels,xlim=(0,1740),xlabel='Fully feasible conditions / 1620')
    for i,r in enumerate(rr):aa[0].text(float(r['count'])+22,i,str(r['count']),va='center',fontsize=8.5)
    rr=read('Fig4b_first_failure.csv');labels=['Instantaneous field','Daylight interval','Prediction model','Error allowances','Manual / hold']
    aa[1].barh(np.arange(5),vals(rr,'share_pct'),color=C['grey'],height=.58)
    aa[1].set(yticks=np.arange(5),yticklabels=labels,xlim=(0,36),xlabel='First-failure share (%)')
    for i,r in enumerate(rr):aa[1].text(float(r['share_pct'])+.6,i,f"{float(r['share_pct']):.1f}",va='center',fontsize=8.5)
    aa[0].set_title('(a) Common 81-point lookup');aa[1].set_title('(b) 72 condition-days')
    for ax in aa:grid(ax,'x');ax.margins(y=.10);ax.invert_yaxis()
    save(fig,'Fig4_feasibility',out)

def point_scatter(ax,rows):
    for acc,col,mark in [('1',C['teal'],'o'),('0',C['red'],'x')]:
        rr=subset(rows,'both_accepted',acc)
        # Plot one symbol per exact coordinate and class; multiplicities are labelled.
        unique={ (r['energy_difference_kWh_day'],r['joint_shortfall_difference_pp']):r for r in rr }
        rr=list(unique.values());kw={'facecolors':'white','edgecolors':col,'linewidths':.9} if acc=='1' else {'color':col,'linewidths':1.0}
        ax.scatter(vals(rr,'energy_difference_kWh_day'),vals(rr,'joint_shortfall_difference_pp'),s=23,marker=mark,zorder=4,**kw)

def fig5(out):
    fig=canvas(4.6);gs=fig.add_gridspec(3,2,height_ratios=[1,.74,.12])
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
    foot=fig.add_subplot(gs[2,:]);foot.axis('off');foot._legend_only=True
    rows=read('Fig5a_point_differences.csv');point_scatter(a,rows);point_scatter(b,rows)
    zero=[r for r in rows if float(r['energy_difference_kWh_day'])==0 and float(r['joint_shortfall_difference_pp'])==0]
    assert len(zero)==26 and sum(r['both_accepted']=='1' for r in zero)==23
    for ax in [a,b]:
        for ln in [ax.axhline(0,color=C['grey'],lw=.55),ax.axvline(0,color=C['grey'],lw=.55)]:ln._audit_guide=True
        grid(ax);ax.set_xlabel(r'$\Delta W$ (kWh/day)');ax.set_ylabel(r'$\Delta J_{\mathrm{auto}}$ (pp)')
    for value in [-.2,.2]:a.axhline(value,color=C['orange'],ls='--',lw=.9)._audit_guide=True
    for value in [-.05,.05]:a.axvline(value,color=C['orange'],ls='--',lw=.9)._audit_guide=True
    a.set(xlim=(-.25,.075),ylim=(-1.55,.57),xticks=[-.2,-.1,0],title='(a) All 36 conditions')
    b.set(xlim=(-.093,.014),ylim=(-.0030,.0040),xticks=[-.08,-.04,0],yticks=[-.002,0,.002,.004],title='(b) Near-zero differences')
    a_positions={'R02':(-.155,.07),'R10':(-.20,-.75),'R15':(-.13,-.36),'R30':(-.12,-1.36),'R28':(-.12,.43)}
    b_offsets={'R02':(7,10),'R19':(9,5),'R08':(-5,13),'R06':(-35,10),'R11':(-34,-16),'R09':(4,-22)}
    for ax,offsets,coords in [(a,a_positions,'data'),(b,b_offsets,'offset points')]:
        for r in rows:
            if r['case'] in offsets:
                ax.annotate(r['case'],(float(r['energy_difference_kWh_day']),float(r['joint_shortfall_difference_pp'])),
                    xytext=offsets[r['case']],textcoords=coords,fontsize=8.3,
                    color=C['red'] if r['both_accepted']=='0' else C['ink'],
                    arrowprops={'arrowstyle':'-','lw':.5,'color':C['grey'],'shrinkB':4})
    b.annotate('26 at (0, 0)\n23 accepted; 3 failed',xy=(0,0),xytext=(-.052,-.00115),ha='center',va='center',
        fontsize=8,linespacing=1.2,arrowprops={'arrowstyle':'-','lw':.5,'color':C['grey'],'shrinkB':4})
    rr=read('Fig5b_classification.csv');labels=['Below screening scales','Energy candidate','Acceptance failed','Quality deterioration','Quality candidate']
    c.barh(np.arange(5),vals(rr,'count'),color=[C['grey'],C['blue'],C['red'],C['orange'],C['teal']],height=.58)
    c.set(yticks=np.arange(5),yticklabels=labels,xlabel='Number of conditions',title='(c) Nominal screening outcomes',xlim=(0,30),xticks=[0,10,20,30]);c.invert_yaxis()
    for i,r in enumerate(rr):c.text(float(r['count'])+.3,i,r['count'],va='center',fontsize=8.5)
    grid(c,'x')
    footer(foot,handles=[Line2D([],[],color=C['teal'],marker='o',mfc='white',ls='none'),
        Line2D([],[],color=C['red'],marker='x',ls='none'),Line2D([],[],color=C['orange'],ls='--')],
        labels=['Both configurations accepted','Acceptance failed','Engineering screening scales'],ncol=3)
    save(fig,'Fig5_point_value',out)

def fig6(out):
    fig=canvas(3.05);gs=fig.add_gridspec(2,2,height_ratios=[1,.18]);aa=[fig.add_subplot(gs[0,i]) for i in range(2)]
    foot=fig.add_subplot(gs[1,:]);foot.axis('off');foot._legend_only=True;rows=read('Fig6_binning.csv')
    spans=[]
    for p,label,col,mark,ls in [('PU','Joint enumeration (PU)',C['teal'],'o','-'),('DU','Zonal union with repair (DZU)',C['blue'],'s','--')]:
        rr=sorted(subset(rows,'policy',p),key=lambda r:int(r['bins']))
        for ax,field in [(aa[0],'energy_kWh_day'),(aa[1],'joint_shortfall_pct')]:
            ln=line(ax,vals(rr,'bins'),vals(rr,field),label,col,mark,ls)
            if p=='DU':ln.set_markerfacecolor('white');ln.set_markersize(5.5);ln.set_zorder(3)
            else:ln.set_zorder(4)
        spans.append(np.ptp(vals(rr,'joint_shortfall_pct')))
    aa[0].set(ylabel='Daily energy (kWh/day)',title='(a) Common-reference energy')
    aa[1].set(ylabel=r'Automatic-reading joint shortfall, $J_{\mathrm{auto}}$ (%)',title='(b) Absolute joint shortfall',ylim=(6.30,6.50),yticks=[6.30,6.35,6.40,6.45,6.50])
    aa[1].text(.5,.91,f'Maximum within-policy range: {max(spans):.3f} pp',transform=aa[1].transAxes,ha='center',fontsize=8)
    for ax in aa:ax.set(xticks=[3,6,12],xlabel='Number of daylight bins');grid(ax,'y')
    footer(foot,aa[0],ncol=2);save(fig,'Fig6_baselines_and_bins',out)

def fig7(out):
    fig=canvas(3.7);gs=fig.add_gridspec(2,2,height_ratios=[1,.20],width_ratios=[1,1.2])
    aa=[fig.add_subplot(gs[0,i]) for i in range(2)];foot=fig.add_subplot(gs[1,:]);foot.axis('off');foot._legend_only=True
    rr=read('Fig7_paired_bounds.csv');status={r['case']:r['both_accepted']=='1' for r in read('Fig5a_point_differences.csv')}
    for k,ax in enumerate(aa):
        lim=31 if k==0 else 3
        for i,r in enumerate(rr):
            lo,hi,nom=[float(r[x]) for x in ['lower_pp','upper_pp','nominal_difference_pp']]
            passed=status[r['case']];col=C['blue'] if passed else C['red']
            ax.plot([max(lo,-lim),min(hi,lim)],[i,i],color=col,lw=1.05,ls='-' if passed else '--',zorder=2)
            for endpoint in [lo,hi]:
                if -lim<=endpoint<=lim:ax.plot([endpoint,endpoint],[i-.10,i+.10],color=col,lw=.8,zorder=2)
            if lo < -lim:ax.annotate('',xy=(-lim,i),xytext=(-lim+.35,i),arrowprops={'arrowstyle':'->','color':col,'lw':1})
            if hi > lim:ax.annotate('',xy=(lim,i),xytext=(lim-.35,i),arrowprops={'arrowstyle':'->','color':col,'lw':1})
            kw={'facecolors':'white','edgecolors':C['teal'],'linewidths':.9} if passed else {'color':C['red'],'linewidths':1.0}
            ax.scatter([nom],[i],marker='o' if passed else 'x',s=23,zorder=4,**kw)
        ax.axvline(0,color=C['grey'],lw=.55)
        if k==1:
            for value in [-.2,.2]:ax.axvline(value,color=C['orange'],ls='--',lw=.85)
        ax.set(xlim=(-lim,lim),ylim=(len(rr)-.5,-.7),yticks=np.arange(len(rr)),
            yticklabels=[r['case'] for r in rr],xlabel=r'$\Delta J_{\mathrm{auto}}$: 24 minus 12 points (pp)');grid(ax,'x')
        for text,r in zip(ax.get_yticklabels(),rr):
            if not status[r['case']]:text.set_color(C['red'])
    aa[0].set(title='(a) Complete conditional bounds',xticks=[-30,-15,0,15,30])
    aa[1].set(title='(b) Detail within −3 to 3 pp',xticks=[-3,-2,-1,0,1,2,3])
    footer(foot,handles=[Line2D([],[],color=C['teal'],marker='o',mfc='white',ls='none'),Line2D([],[],color=C['blue']),
        Line2D([],[],color=C['red'],marker='x',ls='--'),Line2D([],[],color=C['orange'],ls='--')],
        labels=['Nominal difference: accepted','Conditional outer bounds','Research acceptance failed','Quality scales: −0.2 and +0.2 pp'],ncol=2)
    save(fig,'Fig7_paired_quality_observability',out)

def figs1(out):
    fig=canvas(5.4);gs=fig.add_gridspec(2,1,height_ratios=[1,.08]);ax=fig.add_subplot(gs[0]);foot=fig.add_subplot(gs[1]);foot.axis('off');foot._legend_only=True
    rr=read('FigS1_acceptance.csv');data=np.array([int(r['accepted']) for r in rr]).reshape(36,12)
    fail='#EED2CB';pas='#D6E8E5'
    ax.pcolormesh(np.arange(13)-.5,np.arange(37)-.5,data,cmap=ListedColormap([fail,pas]),vmin=0,vmax=1,
        shading='flat',edgecolors='#FFFFFF',linewidth=.28)
    for i,j in np.argwhere(data==0):ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='none',edgecolor=C['red'],hatch='//',lw=0))
    for x in [3.5,7.5]:ax.axvline(x,color='white',lw=1.5)
    for y in [11.5,23.5]:ax.axhline(y,color='white',lw=1.5)
    ax.set(xlim=(-.5,11.5),ylim=(35.5,-.5),xticks=np.arange(12),xticklabels=[str(n) for l in LAYOUTS for n in [8,12,16,24]],
        yticks=np.arange(36),yticklabels=[f'R{i:02}' for i in range(1,37)],title='Research acceptance at 208 positions')
    ax.tick_params(axis='y',labelsize=8,length=0);ax.tick_params(axis='x',labelsize=8.5,length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    group_axis=ax.secondary_xaxis('bottom');group_axis.set_xticks([1.5,5.5,9.5],labels=[LAYLABEL[l] for l in LAYOUTS])
    group_axis.spines['bottom'].set_position(('outward',18));group_axis.spines['bottom'].set_visible(False);group_axis.tick_params(length=0,labelsize=9,pad=3)
    footer(foot,handles=[Patch(facecolor=pas),Patch(facecolor=fail,edgecolor=C['red'],hatch='//')],labels=['Passed','Failed'],ncol=2)
    save(fig,'FigS1_acceptance_matrix',out)

def figs2(out):
    fig,aa,feet=panels(height=3.45);aa[0].get_subplotspec().get_gridspec().set_height_ratios([1,.25])
    rr=read('FigS2a_optical_RMSE.csv');x=np.arange(2)
    for j,method,col,label,hatch in [(0,'original',C['grey'],'Analytic baseline','//'),(1,'improved',C['teal'],'Improved model','')]:
        values=[float(next(r for r in rr if r['condition']==k and r['method']==method)['mean_RMSE_lx']) for k in ['angular_1p6','window_nonuniform']]
        aa[0].bar(x+(j-.5)*.32,values,width=.29,color=col,label=label,hatch=hatch,edgecolor='white',lw=.5)
    aa[0].set(xticks=x,xticklabels=['Changed angular\ndistribution','Non-uniform\nwindow source'],ylabel='12-point combined RMSE (lx)',title='(a) Three room-size conditions')
    rr=read('FigS2b_meter_precision.csv')
    for kind,label,col,mark,ls in [('original_bound','Declared metering bound',C['orange'],'^','-'),('smaller_bound','Smaller-bound scenario',C['blue'],'s','--')]:
        rows=subset(rr,'series',kind);line(aa[1],vals(rows,'days'),vals(rows,'half_width_kWh'),label,col,mark,ls)
    aa[1].axhline(.05,color=C['orange'],ls='--',lw=.9,label='Energy scale: 0.05 kWh/day')
    aa[1].set(xlabel='Total days across four configurations',ylabel='Interval half-width (kWh/day)',title='(b) SD = 0.04 kWh/day')
    grid(aa[0],'y');grid(aa[1],'y')
    for ax,foot in zip(aa,feet):footer(foot,ax)
    save(fig,'FigS2_independence_and_precision',out)

def figs3(out):
    fig=canvas(3.6);gs=fig.add_gridspec(3,2,height_ratios=[.10,1,.23])
    aa=[fig.add_subplot(gs[1,i]) for i in range(2)];feet=[fig.add_subplot(gs[2,i]) for i in range(2)]
    for i,title in enumerate(['(a) Internal shape change','(b) Screen and full-position condition']):
        header=fig.add_subplot(gs[0,i]);header.axis('off');header._legend_only=True
        header.text(.5,.5,title,ha='center',va='center',fontsize=9.5)
        feet[i].axis('off');feet[i]._legend_only=True
    rr=read('FigS3a_task_points.csv');ax=aa[0]
    cmap=LinearSegmentedColormap.from_list('response_change',['#DFD9CC','#CAA579',C['orange'],C['red']])
    sc=ax.scatter(vals(rr,'x_m'),vals(rr,'y_m'),c=vals(rr,'response_change_pct'),cmap=cmap,s=12,vmin=0,vmax=15,edgecolors='none')
    rows=read('FigS3a_monitored_positions.csv')
    for roles,mark,col,size,label in [(['near_sentinel','far_sentinel'],'*',C['teal'],105,'Near/far sentinels: 0%'),(['reference'],'^',C['blue'],18,'Reference: 0%')]:
        rows2=[r for r in rows if r['role'] in roles]
        kw={'facecolors':'none','edgecolors':col,'linewidths':.8} if mark=='*' else {'color':col,'edgecolors':'white','linewidths':.3}
        ax.scatter(vals(rows2,'x_m'),vals(rows2,'y_m'),marker=mark,s=size,label=label,zorder=4 if mark=='*' else 5,**kw)
    ax.set(xlabel='Depth from window (m)',ylabel='Lateral position (m)');ax.margins(.05);ax.set_aspect('equal');ax.set_anchor('N')
    for sp in ['top','right']:ax.spines[sp].set_visible(False)
    cb=fig.colorbar(sc,ax=ax,fraction=.05,pad=.025);cb.set_label('Daylight-response change (%)',fontsize=8.5);cb.ax.tick_params(labelsize=8)
    rr=read('FigS3b_counterexamples.csv');x=np.arange(len(rr));ax=aa[1]
    ax.bar(x,vals(rr,'max_task_change_pct'),color=[C['teal'],C['red'],C['red'],C['red']],hatch=['','//','',''],width=.55,edgecolor='white',lw=.6)
    ax.axhline(5,color=C['grey'],ls='--',lw=.85,label='Full-position bound: 5%')
    for i,r in enumerate(rr):ax.text(i,float(r['max_task_change_pct'])+.5,'Sentinel\nscreen\npassed' if r['screen_pass']=='1' else 'Sentinel\nscreen\nfailed',ha='center',fontsize=8,linespacing=1.08)
    ax.set(xticks=x,xticklabels=['Bounded\nchange','Detectable\nat sentinels','Hidden\ninternal\nchange','Returning\ninternal\nchange'],ylabel='Maximum task-point change (%)',ylim=(0,19.2))
    grid(ax,'y');lg=footer(feet[0],aa[0]);lg.set_loc('upper center');lg.set_bbox_to_anchor((.5,1.0))
    feet[0].text(.5,.02,'Reference and near sentinel share a position',ha='center',va='bottom',transform=feet[0].transAxes,fontsize=8)
    footer(feet[1],handles=[Patch(facecolor=C['teal']),Patch(facecolor=C['red']),Line2D([],[],color=C['grey'],ls='--')],
        labels=['Full-position condition met','Full-position condition not met','Full-position bound: 5%'])
    save(fig,'FigS3_sentinel_full_plane_counterexample',out)

def figs4(out):
    fig,aa,feet=panels(height=3.4);rows=read('FigS4a_reference_trace.csv')
    aa[0].plot(vals(rows,'time_s'),vals(rows,'camera_reference_lx'),color=C['teal'],label='Camera reference (10 s mean)')
    aa[0].plot(vals(rows,'time_s'),vals(rows,'linear_interpolation_lx'),color=C['orange'],ls='--',label='Linear endpoint interpolation')
    aa[0].set(xlabel='Time within group (s)',ylabel='Reference illuminance proxy (lx)',title='(a) Returning internal peak');grid(aa[0],'y')
    rr=read('FigS4b_session_screens.csv');x=np.arange(5)
    aa[1].bar(x-.16,vals(rr,'internal_residual_pct'),width=.29,color=C['blue'],label='Internal-reference residual')
    aa[1].bar(x+.16,vals(rr,'sentinel_change_pct'),width=.29,color=C['orange'],hatch='//',edgecolor='white',lw=.4,label='Spatial sentinel change')
    aa[1].axhline(3.5,color=C['blue'],ls=':',lw=.9);aa[1].axhline(2,color=C['orange'],ls='--',lw=.9)
    aa[1].set(xticks=x,xticklabels=['Steady','Returning\npeak','Shape\nchange','Repeat A','Repeat B'],ylabel='Screening statistic (%)',title='(b) Temporal and spatial checks')
    grid(aa[1],'y')
    for ax,foot in zip(aa,feet):footer(foot,ax)
    feet[1].text(.5,-.08,'Limits: internal residual 3.5%; sentinel change 2%',ha='center',va='top',transform=feet[1].transAxes,fontsize=7.8)
    save(fig,'FigS4_sequential_qualification',out)

FUNCTIONS=[fig1,fig2,fig3,fig4,fig5,fig6,fig7,figs1,figs2,figs3,figs4]
def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE.parent/'figures_for_submission_en');ap.add_argument('--font-dir');ap.add_argument('--figure',type=int,choices=range(1,12));a=ap.parse_args(argv)
    a.output=a.output.resolve()
    if a.output.exists():ap.error('Choose a new output directory; existing files are never overwritten.')
    if a.output.is_relative_to(HERE.parent.resolve()):ap.error('Keep new figures outside the archived Study directory.')
    font=setup(a.font_dir)
    for i,fn in enumerate(FUNCTIONS,1):
        if a.figure is None or a.figure==i:fn(a.output);print('Drawn',fn.__name__,flush=True)
    (a.output/'figure_style_audit.json').write_text(json.dumps({'style':STYLE['version'],'font_family':'Times New Roman','figures':AUDIT},indent=2),encoding='utf-8')
if __name__=='__main__':main()
