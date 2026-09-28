from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
p=p.filter(pl.col('median_ms').is_not_null())
x=p.filter(pl.col('k')==100).group_by(['backend','filter_kind','mode','bs','p_n_lists','p_n_probe']).agg(pl.col('median_ms').median(),pl.col('peak_fwd_mib').max())
print(x.filter(pl.col('backend')=='official').pivot(on='p_n_probe',index=['filter_kind','bs','p_n_lists'],values='median_ms').sort(['filter_kind','bs','p_n_lists']))
print(x.filter(pl.col('mode')=='eager').pivot(on='p_n_probe',index=['backend','filter_kind','bs','p_n_lists'],values='peak_fwd_mib').sort(['backend','filter_kind','bs','p_n_lists']))
# np4 vs np8 triton graph per sweep/kind
y=p.filter((pl.col('backend')=='triton')&(pl.col('p_n_lists')==1664)&(pl.col('p_n_probe').is_in([4,8]))).group_by(['mode','bs','k','p_n_probe']).agg(pl.col('median_ms').median()).pivot(on='p_n_probe',index=['mode','bs','k'],values='median_ms').sort(['mode','bs','k'])
print(y)
