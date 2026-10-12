from load import load
import polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
print(c.filter((pl.col('backend')=='official')&(pl.col('parity')=='reference')).select(['line','filter_kind','sweep','seed','p_n_lists','p_n_probe']).head(50))
o=c.filter(pl.col('backend')=='official')
print(o.select(['line','filter_kind','sweep','seed','p_n_lists','p_n_probe','parity']).head(15))
print(o.filter(pl.col('parity')=='vs_triton').select(pl.col('jaccard_vs_first@100','jaccard_vs_first@400','score_max_abs_diff').describe()))
