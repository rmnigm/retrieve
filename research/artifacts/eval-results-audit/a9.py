from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
C=[]
for ds in ['goodreads-d128','arxiv-d128','yfcc10m-d192','pubmed-d768']:
    c,p=load(f'snap/filter/{ds}.jsonl'); C.append(c)
c=pl.concat(C,how='diagonal_relaxed').filter(pl.col('status')=='ok')
cd,_=load('snap/deep/arxiv-d128.jsonl')
for name,x in [('filter',c),('deep',cd)]:
    print(name); print(x.group_by(['dataset','algo','backend','filter_kind']).agg(pl.col('bloom_fp_rate').null_count().alias('null'),pl.col('bloom_fp_rate').max(),pl.len()).sort(['dataset','algo','backend','filter_kind']))
key=['dataset','filter_kind','sweep','seed','params']
pm=c.filter((pl.col('dataset')=='pubmed'))
t=pm.filter(pl.col('backend')=='triton').select(key+['algo','oracle_recall@100','heldout_recall@100','parity','jaccard_vs_first@100','score_max_abs_diff'])
print(pm.select(key+['algo','backend','oracle_recall@100','heldout_recall@100','n_kept','parity','jaccard_vs_first@100','score_max_abs_diff','index_mib']).sort(['algo','filter_kind','sweep','params','backend']))
