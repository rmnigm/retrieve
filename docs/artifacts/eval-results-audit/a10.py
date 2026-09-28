import json, polars as pl
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(30); pl.Config.set_tbl_width_chars(250)
rows=[]
for ds in ['goodreads-d128','arxiv-d128','yfcc10m-d192','pubmed-d768']:
    for l in open(f'snap/filter/{ds}.jsonl'):
        r=json.loads(l)
        if r['status']!='ok': continue
        q=r['quality']
        rows.append(dict(ds=r['dataset'],algo=r['algo'],be=r['backend'],fk=r['filter_kind'],sw=r['sweep'],seed=r['seed'],np=(r['params'] or {}).get('n_probe'),par=q['parity'],j=q.get('jaccard_vs_first@100'),d=q.get('score_max_abs_diff'),nq=r['n_queries'],nk=r['n_kept'],nh=r.get('n_queries_heldout'),nt=r.get('n_targets_in_filter'),no=r.get('n_queries_oracle'),h=q['heldout']['recall@100'],o=q['oracle']['recall@100'],pr=r['pass_rate']))
df=pl.DataFrame(rows)
print(df.filter(pl.col('par').str.starts_with('vs')).group_by(['ds','fk']).agg(pl.col('d').min().alias('dmin'),pl.col('d').max().alias('dmax'),pl.col('j').min().alias('jmin'),pl.len()).sort(['ds','fk']))
print(df.filter(pl.col('par').str.starts_with('vs')&(pl.col('d')>0.001)).select(['ds','fk','sw','seed','np','j','d']))
print(df.filter(pl.col('algo')=='linr_v1_filter_mask').select(['ds','fk','sw','seed','nq','nk','nh','nt','no','h','o','pr']).sort(['ds','sw','fk']))
