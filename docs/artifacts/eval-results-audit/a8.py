from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
C=[];P=[]
for ds in ['goodreads-d128','arxiv-d128','yfcc10m-d192','pubmed-d768']:
    c,p=load(f'snap/filter/{ds}.jsonl'); C.append(c); P.append(p)
c=pl.concat(C,how='diagonal_relaxed'); p=pl.concat(P,how='diagonal_relaxed')
c=c.filter(pl.col('status')=='ok')
key=['dataset','filter_kind','sweep','seed']
v1=c.filter(pl.col('algo')=='linr_v1_filter_mask').select(key+[pl.col('oracle_recall@100').alias('v1r'),pl.col('oracle_recall@1000').alias('v1r1k'),pl.col('heldout_recall@100').alias('v1h'),pl.col('oracle_ndcg@100').alias('v1n')])
v2=c.filter(pl.col('algo')=='linr_v2').select(key+[pl.col('oracle_recall@100').alias('v2r'),pl.col('oracle_recall@1000').alias('v2r1k'),pl.col('heldout_recall@100').alias('v2h'),pl.col('oracle_ndcg@100').alias('v2n')])
j=v1.join(v2,on=key).with_columns((pl.col('v1r')-pl.col('v2r')).abs().alias('dr'),(pl.col('v1r1k')-pl.col('v2r1k')).abs().alias('dr1k'),(pl.col('v1n')-pl.col('v2n')).abs().alias('dn'))
print(j.group_by('dataset').agg(pl.col('dr').max(),pl.col('dr1k').max(),pl.col('dn').max(),pl.len(), pl.col('v1r').min()))
# seed spread
s=c.filter(pl.col('seed').is_in([0,1,2])).group_by(['dataset','algo','backend','filter_kind','sweep','params']).agg(pl.col('seed').n_unique().alias('ns'),pl.col('oracle_recall@100').min().alias('mn'),pl.col('oracle_recall@100').max().alias('mx')).filter(pl.col('ns')>1).with_columns((pl.col('mx')-pl.col('mn')).alias('rng'))
print(s.sort('rng',descending=True).head(25))
# linr_v3 & others quality summary
print(c.group_by(['dataset','algo','backend']).agg(pl.col('oracle_recall@100').mean(),pl.col('oracle_recall@1000').mean(),pl.col('heldout_recall@100').mean(),pl.col('index_mib').mean(),pl.col('filter_mib').max(),pl.col('bloom_fp_rate').max(),pl.col('bloom_fp_rate').null_count().alias('fpnull'),pl.len()).sort(['dataset','algo']))
num=[k for k,t in zip(c.columns,c.dtypes) if t==pl.Float64]
for k in num:
    s_=c[k]; 
    if s_.is_nan().sum() or (s_<0).sum(): print('NAN/NEG',k)
pp=p.filter(pl.col('median_ms').is_not_null())
for k in ['median_ms','p99_ms','qps','peak_fwd_mib','spread','min_ms']:
    print(k, pp[k].min(), pp[k].max(), pp[k].is_nan().sum())
print(p.filter(pl.col('median_ms').is_null()).group_by(['algo','backend','mode','reason']).len())
