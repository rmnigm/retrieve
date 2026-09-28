from load import load
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
print(c.group_by(['filter_kind','sweep','backend','seed']).len().sort(['filter_kind','sweep','backend','seed']))
# nulls / nan / zeros
num=[k for k,t in zip(c.columns,c.dtypes) if t in (pl.Float64,pl.Int64)]
for k in num:
    s=c[k]
    nn=s.null_count(); nan=s.is_nan().sum() if s.dtype==pl.Float64 else 0; z=(s==0).sum(); neg=(s<0).sum()
    if nn or nan or z or neg: print(k,'null',nn,'nan',nan,'zero',z,'neg',neg)
nump=['median_ms','mean_ms','p99_ms','min_ms','qps','host_gap_ms','peak_fwd_mib','sm_mhz','spread','iqr_ms']
for k in nump:
    s=p[k]; print(k,'null',s.null_count(),'nan',s.is_nan().sum(),'zero',(s==0).sum(),'neg',(s<0).sum(), 'min',s.min(),'max',s.max())
print(p.filter(pl.col('median_ms').is_null()).group_by(['backend','mode','reason']).len())
print(c.select(['parity']).group_by('parity').len())
