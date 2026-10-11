from load import load
import polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
c,p=load('snap/deep/arxiv-d128.jsonl')
p=p.filter(pl.col('median_ms').is_not_null())
idx=['backend','filter_kind','sweep','seed','p_n_lists','p_n_probe','k','mode']
b=p.filter(pl.col('bs')==1).select(idx+[pl.col('median_ms').alias('m1')]).join(p.filter(pl.col('bs')==16).select(idx+[pl.col('median_ms').alias('m16')]),on=idx).with_columns((pl.col('m16')/pl.col('m1')).alias('ratio'))
print(b.group_by(['backend','mode']).agg(pl.col('ratio').max().alias('max'),pl.col('ratio').median().alias('med'),(pl.col('ratio')>8).sum().alias('gt8'),(pl.col('ratio')>=16).sum().alias('viol')))
print(b.sort('ratio',descending=True).head(6))
# graph vs eager speedup
e=p.filter(pl.col('backend')=='triton')
idx2=['filter_kind','sweep','seed','p_n_lists','p_n_probe','k','bs']
g=e.filter(pl.col('mode')=='eager').select(idx2+[pl.col('median_ms').alias('e')]).join(e.filter(pl.col('mode')=='graph').select(idx2+[pl.col('median_ms').alias('g')]),on=idx2).with_columns((pl.col('g')/pl.col('e')).alias('ge'))
print(g.group_by(['bs','p_n_probe']).agg(pl.col("ge").median().alias("med"),pl.col("ge").max().alias("mx")).sort(['bs','p_n_probe']))
print('graph slower than eager:',g.filter(pl.col('ge')>1).height)
