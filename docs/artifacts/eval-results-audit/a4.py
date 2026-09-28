from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
# recall monotone in n_probe
g=['backend','filter_kind','sweep','seed','p_n_lists']
for m in ['oracle_recall@100','oracle_recall@400','heldout_recall@100']:
    d=c.sort(g+['p_n_probe']).with_columns(pl.col(m).diff().over(g).alias('dd'))
    bad=d.filter(pl.col('dd')< -1e-9)
    print(m,'non-monotone steps:',bad.height); 
    if bad.height: print(bad.select(g+['p_n_probe',m,'dd']).head(20))
# summary curve (triton, median across seeds, clause)
t=c.filter(pl.col('backend')=='triton')
print(t.group_by(['filter_kind','p_n_lists','p_n_probe']).agg(pl.col('oracle_recall@100').mean().alias('r100'),pl.col('oracle_recall@400').mean().alias('r400'),pl.col('heldout_recall@100').mean().alias('h100'),pl.col('pass_rate').mean()).sort(['filter_kind','p_n_lists','p_n_probe']))
print(t.group_by(['sweep']).agg(pl.col('pass_rate').mean(), pl.col('n_kept').mean()).sort('sweep'))
