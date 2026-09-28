from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
p=p.filter(pl.col('median_ms').is_not_null())
g=['backend','filter_kind','sweep','seed','p_n_lists','k','bs','mode']
d=p.sort(g+['p_n_probe']).with_columns((pl.col('median_ms')/pl.col('median_ms').shift(1).over(g)).alias('ratio'))
bad=d.filter(pl.col('ratio')<0.95)
print('latency drops >5% when n_probe increases:',bad.height,'of',d.filter(pl.col('ratio').is_not_null()).height)
print(bad.group_by(['backend','mode','bs','p_n_lists','p_n_probe']).len().sort('len',descending=True).head(30))
print(bad.sort('ratio').select(g+['p_n_probe','median_ms','ratio','unstable','spread','sm_mhz']).head(20))
# latency curves mean
print(p.filter(pl.col('k')==100).group_by(['backend','mode','bs','p_n_lists','p_n_probe']).agg(pl.col('median_ms').median()).pivot(on='p_n_probe',index=['backend','mode','bs','p_n_lists'],values='median_ms').sort(['backend','mode','bs','p_n_lists']))
