import polars as pl
T='/scratch/wt/export-r4/results/tables/'
r=lambda n: pl.read_csv(T+n+'.csv',infer_schema_length=None)
# Fig 1
v=pl.concat([r('linr_v2_over_v1'),r('linr_v2_over_v1_batch64')]).filter(~pl.col('dataset').str.starts_with('arxiv-corr'))
print('F1 lines',v.select('dataset','dim','batch').unique().height,'points',v.height,'unstable',v['timing_unstable'].sum())
for b in (1,16,64):
    x=v.filter(pl.col('batch')==b); print(' batch',b,'lines',x.select('dataset','dim').unique().height,'datasets',sorted(x['dataset'].unique()), 'ratio range',x['v2_over_v1'].min(),x['v2_over_v1'].max())
    n=x.filter(pl.col('pass_rate').is_between(0.09,0.11)&~pl.col('timing_unstable')); print('   near10% stable',n['v2_over_v1'].min(),n['v2_over_v1'].max())
    lo=x.sort('pass_rate').group_by('dataset','dim').first(); print('   lowest-rate ratio',lo['v2_over_v1'].min(),lo['v2_over_v1'].max())
nv=r('linr_native_v2_over_v1').filter((pl.col('batch')==1)&(pl.col('mode')=='graph')); print(' native diamonds',nv['native_v2_over_native_v1'].to_list())
# Fig 2
h=r('head_to_head_probe_budgets').filter(pl.col('k')==100)
print('F2 settings',h.height, h.group_by('outcome').len().sort('outcome').rows())
q=h.filter(pl.col('outcome')=='within_quarter'); print(' within quarter',q.height,'faster',(q['silvertorch_over_linr']<1).sum(),'median',q['silvertorch_over_linr'].median(), 'verdicts',q.group_by('verdict').len().sort('verdict').rows())
for t in (0.9,0.95):
    x=h.filter(pl.col('recall_target')==t); qq=x.filter(pl.col('outcome')=='within_quarter')
    print(' target',t,'n',x.height,'quarter',qq.height,'faster',(qq['silvertorch_over_linr']<1).sum(),'deeper',x.filter(pl.col('outcome')=='deeper').height,'not',x.filter(pl.col('outcome')=='not_reached').height, 'range quarter',qq['silvertorch_over_linr'].min(),qq['silvertorch_over_linr'].max())
print(' losses within quarter',q.filter(pl.col('silvertorch_over_linr')>=1).select('dataset','filter','batch','recall_target','silvertorch_over_linr').rows())
print(' synthetic not reached', h.filter((pl.col('filter_type')=='synthetic')&(pl.col('outcome')=='not_reached')).select('dataset','filter','batch','recall_target').rows())
print(' synthetic deeper', h.filter((pl.col('filter_type')=='synthetic')&(pl.col('outcome')=='deeper')).select('dataset','filter','batch','recall_target','silvertorch_over_linr','probe_fraction').rows())
s=h.filter((pl.col('filter_type')=='synthetic')&(pl.col('recall_target')==0.9)&(pl.col('outcome')!='not_reached')); print(' synth 0.90 any depth', s.height,'faster',(s['silvertorch_over_linr']<1).sum(),'speedups',1/s['silvertorch_over_linr'].max(),1/s['silvertorch_over_linr'].min())
# Fig 3a
a=r('head_to_head_probe_budgets').filter(pl.col('dataset').str.contains(r'^laion\d+m$')&(pl.col('recall_target')==0.9)&(pl.col('k')==100))
print('F3a points',a.filter(pl.col('outcome')!='not_reached').height,'not reached',a.filter(pl.col('outcome')=='not_reached').height,'deeper',a.filter(pl.col('outcome')=='deeper').height)
for f in ('tags4','c0_domain'):
  for d in ('laion1m','laion30m'):
    x=a.filter((pl.col('filter')==f)&(pl.col('dataset')==d)); print('  ',f,d,'speed-up',(1/x['silvertorch_over_linr']).round(2).to_list(),x['outcome'].to_list())
# Fig 3b
c=r('codesign_pairs').filter(pl.col('dataset').str.starts_with('laion')); st=c.filter(~pl.col('timing_unstable'))
print('F3b pairs',c.height,'stable drawn',st.height,'dropped',c.height-st.height,'range',st['whole_index_over_searched_clusters'].min(),st['whole_index_over_searched_clusters'].max())
# Fig 4
p=r('probe_sweep').filter((pl.col('n_lists')==4096)&pl.col('filter').is_in(['p1','p01','p001','p0001']))
print('F4 points',p.height,'unstable',p['timing_unstable'].sum(),'v1',p['v1_ms'].min(),p['v1_ms'].max())
for f in ['p1','p01','p001','p0001']:
    x=p.filter(pl.col('filter')==f).sort('n_probe'); print('  ',f,'probes',x['n_probe'].min(),x['n_probe'].max(),'recall',x['silvertorch_recall'][0],x['silvertorch_recall'][-1],'ms',x['silvertorch_ms'][0],x['silvertorch_ms'][-1],'unstable at',x.filter(pl.col('timing_unstable'))['n_probe'].to_list())
