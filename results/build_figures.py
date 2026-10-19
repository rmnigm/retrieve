"""The paper's Figures 1-3 from tables/, written to figures/ (PDF and PNG). Run from results/ with no arguments.

F1-F2: hollow = deeper/rescaled; + = flagged timing. F3: batch-16 heatmap of the head-to-head selections at full
precision, recomputed from records/ with build_tables.py (the CSV rounds the times).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter,NullFormatter
import polars as pl,math
from matplotlib.colors import TwoSlopeNorm,LinearSegmentedColormap
ROOT=Path(__file__).resolve().parent; T=ROOT/'tables'; O=ROOT/'figures'
plt.rcParams.update({'font.size':10.5,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.22,'legend.frameon':False,'pdf.fonttype':42})
COL={1:'#2766b0',16:'#c75a13',64:'#13816b'}; MARK={1:'o',16:'s',64:'^'}
def read(n):return pl.read_csv(T/f'{n}.csv',infer_schema_length=None)
def save(fig,n):
 for ext in ['pdf','png']:fig.savefig(O/f'{n}.{ext}',bbox_inches='tight',dpi=450)
 plt.close(fig)
def f1():
 x=pl.concat([read('linr_v2_over_v1'),read('linr_v2_over_v1_batch64')]).filter(~pl.col('dataset').str.starts_with('arxiv-corr'))
 cats=sorted(x.select('dataset','dim').unique().iter_rows())
 colors=plt.get_cmap('tab10').colors
 fig,axes=plt.subplots(1,3,figsize=(5.8,2.65),sharey=True)
 names={'arxiv-synth':'arXiv','goodreads-synth':'Goodreads','pubmed-synth':'PubMed'}
 for ax,bs in zip(axes,[1,16,64]):
  for i,(ds,d) in enumerate(cats):
   g=x.filter((pl.col('dataset')==ds)&(pl.col('dim')==d)&(pl.col('batch')==bs)).sort('pass_rate')
   if not g.height:continue
   label=names.get(ds,ds.replace('-synth','').replace('laion','LAION ').replace('m','M'))+f' {d}d'
   ax.plot(g['pass_rate']*100,1/g['v2_over_v1'],color=colors[i],marker='s' if ds=='arxiv-synth' and d==256 else 'o',ls='--' if ds=='arxiv-synth' and d==256 else '-',ms=2.5,lw=1,label=label)
   u=g.filter(pl.col('timing_unstable'));ax.plot(u['pass_rate']*100,1/u['v2_over_v1'],'+',color='black',ms=5)
  ax.axhline(1,color='black',lw=.7);ax.set(xscale='log',yscale='log',xlabel='pass rate (%)',title=f'Batch {bs}',ylim=(.02,14))
  ax.set_xticks([.1,1,10,100],['0.1','1','10','100']);ax.set_yticks([.03,.1,1,10],['0.03','0.1','1','10']);ax.yaxis.set_minor_formatter(NullFormatter())
 n=read('linr_native_v2_over_v1').filter((pl.col('batch')==1)&(pl.col('mode')=='graph'))
 axes[0].plot(n['pass_rate']*100,1/n['native_v2_over_native_v1'],'D',ms=4,color='black')
 axes[0].plot(10.97,1/3.042,'*',ms=9,color='black');axes[0].plot(10.97,1/1.095,'*',ms=9,color='#777777')
 axes[0].set_ylabel('V2 speed-up over V1')
 handles,labels=axes[0].get_legend_handles_labels()
 handles += [Line2D([],[],marker='*',color=c,ls='',ms=8) for c in ['black','#777777']]+[Line2D([],[],marker='D',color='black',ls='',ms=4)]
 labels += ['Original PyTorch (b1)','Original TensorFlow (b1)','Goodreads native (b1)']
 fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=8.7,bbox_to_anchor=(.5,-.27),columnspacing=1.0)

 fig.tight_layout();save(fig,'F1')
def f2():
 x=read('probe_sweep').filter(pl.col('n_lists')==4096)
 fig,ax=plt.subplots(figsize=(5.5,2.9))
 for f,col,m,ls in zip(['p1','p01','p001','p0001'],['#2766b0','#c75a13','#13816b','#9752a1'],['o','s','^','D'],['-','--',':','-.']):
  g=x.filter(pl.col('filter')==f).sort('n_probe')
  ax.plot(g['silvertorch_ms'],g['silvertorch_recall'],color=col,marker=m,ms=3,ls=ls,label={'p1':'no filter','p01':'10%','p001':'1%','p0001':'0.1%'}[f])
  for r in g.iter_rows(named=True):
   if r['n_probe'] in [24,1024]:
    offsets = {'p1':(4,-18),'p01':(0,15),'p001':(0,-22),'p0001':(-16,15)}
    offset = ((-3,9) if f=='p0001' else (-5,-16)) if r['n_probe']==24 else offsets[f]
    ax.annotate(str(r['n_probe']),(r['silvertorch_ms'],r['silvertorch_recall']),xytext=offset,textcoords='offset points',fontsize=10,ha='center',color=col,arrowprops={'arrowstyle':'-','lw':.5,'color':col})
  u=g.filter(pl.col('timing_unstable'));ax.plot(u['silvertorch_ms'],u['silvertorch_recall'],'+',color='black',ms=6)
 ax.axvline(x['v1_ms'].median(),color='black',ls='--',lw=.8,label='LiNR V1 (exact)')
 ax.set(xscale='log',xlabel='adapted SilverTorch latency / batch of 16 (ms)',ylabel='recall@100',ylim=(.27,1.03));ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
 ax.legend(loc='lower right',fontsize=10,ncol=2)
 fig.tight_layout();save(fig,'F2')
def f3():
 import build_tables as bt
 r=bt.load();h=bt.h2h_records(r)
 x=bt.probe_budgets(bt.h2h_picks(h,.25),bt.h2h_picks(h,None),bt.timing_spread(bt.h2h_points(h),bt.linr_seed_spread(r))).filter((pl.col('k')==100)&(pl.col('batch')==16))
 cmap=LinearSegmentedColormap.from_list('d',['#b2182b','#f7f7f7','#2166ac']);cmap.set_bad('#e6e6e6');norm=TwoSlopeNorm(0,-2,4)
 def val(ds,f,t):
  g=x.filter((pl.col('dataset')==ds)&(pl.col('filter')==f)&(pl.col('recall_target')==t))
  if g.height==0:return 'skip'
  r=g.row(0,named=True);return None if r['outcome']=='not_reached' else 1/r['silvertorch_over_linr']
 def heat(ax,M,xl,yl,title):
  ax.imshow([[math.nan if v in (None,'skip') else math.log2(v) for v in r] for r in M],cmap=cmap,norm=norm,aspect='auto')
  for i,r in enumerate(M):
   for j,v in enumerate(r):
    if v=='skip':
     ax.add_patch(plt.Rectangle((j-.5,i-.5),1,1,facecolor='#f4f4f4',edgecolor='#e2e2e2',hatch='///',lw=0,zorder=2))
     ax.text(j,i,'NM',ha='center',va='center',fontsize=6.3,color='#999',zorder=3);continue
    ax.text(j,i,'NR' if v is None else f'{v:.1f}' if v<9.95 else f'{v:.0f}',ha='center',va='center',fontsize=6.3,color='#888' if v is None else 'black',zorder=3)
  ax.set_xticks(range(len(xl)),xl,fontsize=6);ax.set_yticks(range(len(yl)),yl,fontsize=6.3);ax.tick_params(length=0);ax.grid(False)
  for s in ax.spines.values():s.set_visible(False)
  ax.set_title(title,fontsize=7,loc='left',pad=3)
 T=[.9,.95];TL=['0.90','0.95']
 real=[('pubmed','c0_mesh','PubMed\nMeSH\n0.02%'),('arxiv','all4','arXiv\n4 cl.\n0.9%'),('arxiv','c0_maincat','arXiv\ncat.\n14%'),('goodreads','c0_genre','Goodreads\ngenre\n33%'),('arxiv','c3_nversions','arXiv\nvers.\n44%')]
 laion=[(f'laion{n}m',f) for f in ['tags4','c0_domain'] for n in [1,3,10,30]]
 cats=[('goodreads','Goodreads 0.8M'),('arxiv','arXiv 3M'),('pubmed','PubMed 10M'),('laion10m','LAION 10M'),('laion30m','LAION 30M')]
 prs=['p0001','p001','p01','p02','p03','p05','p1']
 fig=plt.figure(figsize=(4.8,2.35))
 gs=fig.add_gridspec(2,2,height_ratios=[2,5],width_ratios=[6,8],hspace=.95,wspace=.06)
 a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:])
 heat(a,[[val(d,f,t) for d,f,_ in real] for t in T],[l for *_,l in real],TL,'(a) Real filters, by pass rate');a.tick_params(axis='x',labelsize=4.9)
 heat(b,[[val(d,f,t) for d,f in laion] for t in T],['1M','3M','10M','30M']*2,TL,'(b) LAION, by catalog size');a.set_title(a.get_title('left'),fontsize=7,loc='left',pad=10);b.set_title(b.get_title('left'),fontsize=7,loc='left',pad=10)
 b.axvline(3.5,color='white',lw=2.5);b.set_yticklabels([])
 for xc,lab in [(1.5,'tags, 0.27%'),(5.5,'domain, 0.95%')]:b.text(xc,-.55,lab,ha='center',va='bottom',fontsize=6)
 heat(c,[[val(d+'-synth',p,.9) for p in prs] for d,_ in cats],['0.1%','1%','10%','20%','30%','50%','100%'],[l for _,l in cats],'(c) Synthetic filters at target 0.90, by pass rate and catalog size')
 c.set_xlabel('pass rate',fontsize=6.3,labelpad=2)
 save(fig,'F3')
if __name__=='__main__':
 for f in [f1,f2,f3]:f()
 print('F1–F3 regenerated')
