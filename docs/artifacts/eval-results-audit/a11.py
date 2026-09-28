from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
p=p.filter(pl.col('median_ms').is_not_null())
print('unstable total',p['unstable'].sum(),'of',p.height)
for g in [['backend','mode'],['bs'],['k'],['p_n_probe'],['p_n_lists'],['filter_kind'],['seed']]:
    print(p.group_by(g).agg(pl.col('unstable').mean().alias('frac'),pl.col('unstable').sum().alias('n'),pl.len()).sort(g))
print(p.group_by(['backend','mode','bs']).agg(pl.col('unstable').mean().alias('frac'),pl.col('spread').max()).sort(['backend','mode','bs']))
# throttle: sm_mhz < 1410
print(p.with_columns((pl.col('sm_mhz')<1400).alias('thr')).group_by(['backend','thr']).agg(pl.col('unstable').mean(),pl.len()))
# unstable over time (line index bins)
print(p.with_columns((pl.col('line')//40).alias('bin')).group_by('bin').agg(pl.col('unstable').mean().round(3),pl.col('sm_mhz').min(),pl.len()).sort('bin'))
# record-level unstable vs clocks_drift
import json
dr=[ (json.loads(l)['env'].get('clocks_drift'), json.loads(l).get('unstable')) for l in open('snap/deep/arxiv-d128.jsonl')]
from collections import Counter; print(Counter(dr))
# batch scaling ratio bs16/bs1 median_ms
b=p.filter(pl.col('bs').is_in([1,16])).pivot(on='bs',index=['backend','filter_kind','sweep','seed','p_n_lists','p_n_probe','k','mode'],values='median_ms').with_columns((pl.col('16')/pl.col('1')).alias('r'))
print(b.group_by(['backend','mode']).agg(pl.col('r').max(),pl.col('r').median(),(pl.col('r')>12).sum().alias('gt12')))
print(b.sort('r',descending=True).head(8))
