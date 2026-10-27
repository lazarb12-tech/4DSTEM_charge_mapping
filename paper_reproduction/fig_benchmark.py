"""Figure S4 from tracker_benchmark_results.json (run inside results_v1.1/).
Usage: python ../fig_benchmark.py [median|mean]"""
import json, sys, numpy as np, collections, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.linewidth':0.7,
                     'xtick.major.width':0.6,'ytick.major.width':0.6,'xtick.major.size':3,'ytick.major.size':3,
                     'legend.frameon':False,'axes.spines.top':False,'axes.spines.right':False})
res=json.load(open('tracker_benchmark_results.json'))
g=collections.defaultdict(list)
for r in res: g[r['name']].append(r)
TR=[('ellipse','Ellipse fit','#D55E00','s','-'),
    ('xcorr','Edge cross-correlation','#0072B2','o','-'),
    ('com','Center of mass (reference)','#CC79A7','D','--')]
STAT=sys.argv[1] if len(sys.argv)>1 else 'median'
def stat(names,key):
    out=[]
    for n in names:
        e=np.array([np.hypot(*x[key]) for x in g[n] if np.all(np.isfinite(x[key]))])
        nf=sum(1 for x in g[n] if not np.all(np.isfinite(x[key])))
        if STAT=='median':
            e=np.r_[e,[1e3]*nf]
            out.append(np.median(e))
        else:
            out.append(e.mean() if e.size else np.nan)
    return np.array(out)
fig,axs=plt.subplots(2,4,figsize=(7.2,3.9))
axs=axs.ravel()
def panel(ax,xs,names,xlabel,title,letter,logx=False,keys=('ellipse','xcorr','com'),mark=None):
    for k,lab,c,m,ls in TR:
        if k not in keys: continue
        val=np.clip(stat(names,k),0.02,200)
        ax.plot(xs,val,ls,color=c,marker=m,ms=3.2,lw=1.1,label=lab,mfc='white' if k=='com' else c)
    ax.axhline(0.1,color='0.55',lw=0.6,ls=':')
    ax.set_yscale('log'); ax.set_ylim(0.02,200)
    if logx: ax.set_xscale('log')
    ax.set_xlabel(xlabel); ax.set_title(title,fontsize=8)
    ax.text(-0.30,1.13,letter,transform=ax.transAxes,fontsize=10,fontweight='bold')
    if mark is not None: ax.axvline(mark,color='0.3',lw=0.7,ls='--')
rs=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0,1.5]
panel(axs[0],rs,[f'disc_n1_r{r}' for r in rs],r'$I_g/I_{BF}$','One reflection at the edge','a')
panel(axs[1],rs,[f'disc_n4_r{r}' for r in rs],r'$I_g/I_{BF}$','Two pairs of reflections','b')
rc=[0.1,0.2,0.3,0.5,0.8]
panel(axs[2],rc,[f'coh_n1_r{r}' for r in rc],r'$I_g/I_{BF}$','Coherent overlap','c')
Ds=[0.0,0.3,0.5,0.6,0.7,0.8,0.9,1.0]
panel(axs[3],Ds,[f'band_n2_D{d}' for d in Ds],'Band depth $D$','Dark bands in the disc','d')
Ne=[300,1000,2000,3000,5000,10000,30000,100000,150000]
panel(axs[4],Ne,[f'dose_{n}' for n in Ne],r'Electrons per pattern','Counting noise','e',logx=True,mark=1.5e5)
dR=[0.0,0.5,1.0,1.5,2.0,3.0,5.0]
panel(axs[5],dR,[f'dR_{d}' for d in dR],r'Radius mismatch $\Delta R$ (px)','Template mismatch','f')
for ax in axs[:6]: ax.set_ylabel(('Median' if STAT=='median' else 'Mean')+' |error| (px)')
from matplotlib.lines import Line2D
axs[6].axis('off'); axs[7].axis('off')
h=[Line2D([],[],color=c,marker=m,ls=ls,lw=1.1,ms=3.5,mfc='white' if k=='com' else c,label=lab) for k,lab,c,m,ls in TR]
h.append(Line2D([],[],color='0.55',ls=':',lw=0.8,label='0.1 px'))
axs[6].legend(handles=h,loc='center',fontsize=7.2,bbox_to_anchor=(1.05,0.55))
pass
plt.tight_layout(w_pad=0.6,h_pad=0.9)
plt.savefig('FigS4_tracker_boundaries.png' if STAT=='median' else 'FigS4_tracker_boundaries_mean.png',dpi=300)
print('saved')
