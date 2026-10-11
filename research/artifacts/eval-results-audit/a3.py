from load import load
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
o=c.filter(pl.col('parity')=='vs_triton')
print(o.select(pl.col('jaccard_vs_first@100','jaccard_vs_first@400','score_max_abs_diff')).describe())
print(o.group_by('filter_kind','sweep').agg(pl.col('jaccard_vs_first@100').min().alias('j100min'),pl.col('jaccard_vs_first@400').min().alias('j400min'),pl.col('score_max_abs_diff').max().alias('dmax'),pl.len()).sort('filter_kind','sweep'))
# recall official vs triton matched
key=['filter_kind','sweep','seed','p_n_lists','p_n_probe']
t=c.filter(pl.col('backend')=='triton').select(key+['oracle_recall@100','oracle_recall@400','heldout_recall@100','index_mib','build_s'])
of=c.filter(pl.col('backend')=='official').select(key+['oracle_recall@100','oracle_recall@400','heldout_recall@100','index_mib','build_s','parity','jaccard_vs_first@100'])
j=t.join(of,on=key,suffix='_o').with_columns((pl.col('oracle_recall@100_o')-pl.col('oracle_recall@100')).alias('d100'),(pl.col('oracle_recall@400_o')-pl.col('oracle_recall@400')).alias('d400'))
print(j.select(pl.col('d100','d400')).describe())
print(j.sort('d100').head(8)); print(j.sort('d100').tail(5))
print(j.group_by('filter_kind').agg(pl.col('index_mib').mean(),pl.col('index_mib_o').mean(),pl.col('build_s').mean(),pl.col('build_s_o').mean()))
