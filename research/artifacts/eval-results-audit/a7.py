from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
C=[];P=[]
for ds in ['goodreads-d128','arxiv-d128','yfcc10m-d192','pubmed-d768']:
    c,p=load(f'snap/filter/{ds}.jsonl'); C.append(c); P.append(p)
c=pl.concat(C,how='diagonal_relaxed'); p=pl.concat(P,how='diagonal_relaxed')
print(c.group_by(['dataset','algo','backend','status']).len().sort(['dataset','algo','backend']))
print(c.group_by(['dataset','filter_kind','sweep']).len().sort(['dataset','filter_kind','sweep']))
ok=p.filter(pl.col('median_ms').is_not_null()&(pl.col('k')==100)&(pl.col('mode')=='eager')&(pl.col('algo')=='silvertorch'))
x=ok.group_by(['dataset','filter_kind','backend','bs']).agg(pl.col('median_ms').median(),pl.col('qps').median(),pl.col('peak_fwd_mib').max(),pl.len())
print(x.sort(['dataset','filter_kind','bs','backend']))
