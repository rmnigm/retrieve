"""CPU reduction of the supplementary GPU checks; never launches a GPU job.
Usage: python build_supplementary.py [RAW] [OUT]   (defaults: supplementary/raw, supplementary)
"""
import csv, json, sys
from pathlib import Path
DATA=Path(__file__).resolve().parent/'supplementary'
root=Path(sys.argv[1] if len(sys.argv)>1 else DATA/'raw')
out=Path(sys.argv[2] if len(sys.argv)>2 else DATA)
out.mkdir(parents=True, exist_ok=True)

def read(check, name):
    return [json.loads(s) for s in (root/check/name).read_text().splitlines()]
def write(name, rows):
    with (out/name).open('w') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
ns=[]
for line,r in enumerate(read('check1','native.jsonl'),1):
    for arm in ['native_mask','native_index']:
        for baseline in ['best_graph','triton_v1/eager','triton_v2/eager']:
            key=f'{arm}/vs_{baseline}'
            ns.append(dict(line=line,dataset=r['dataset'],n_items=r['n_items'],batch=r['bs'],k=r['k'],rate=r['rate'],arm=arm,baseline=baseline,ratio=r['ratio'][key],unstable=r['unstable_any']))
write('native.csv',ns)
vs=[]
for line,r in enumerate(read('check2','v3bits.jsonl'),1):
    for key,val in r.get('ratio_to_best_exact', {}).items():
        algo,pool,bs=key.split('/')
        vs.append(dict(line=line,dataset=r['dataset'],dim=r['dim'],n_items=r['n_items'],filter=r['sweep'],pass_rate=r['pass_rate'],queries=r['n_quality_queries'],k=r['k'],algorithm=algo,pool=pool,pool_items=r['pools'][pool],batch=int(bs[2:]),recall=r['recall'][f'{algo}/{pool}'],ratio=val,unstable=r['unstable_any']))
write('v3.csv',vs)
bs=[]
for line,r in enumerate(read('check3','bloom_not.jsonl'),1):
    for key,v in r['recall'].items():
        if key.startswith('bloom/'):
            probe=key.split('/')[1]
            bs.append(dict(line=line,dataset=r['dataset'],filter=r['sweep'],queries=r['n_queries'],probe=probe,fn_rate=r['filter_fn_rate'],not_clause_fn_rate=r['not_clause_fn_rate'],bloom_recall=v,exact_recall=r['recall'][f'exact/{probe}']))
write('not.csv',bs)
print('Native maxima:', {b:max(r['ratio'] for r in ns if r['baseline']==b) for b in ['best_graph','triton_v1/eager','triton_v2/eager']})
quality=[]
for line,r in enumerate(read('check2','v3bits.jsonl'),1):
    for key,recall in r['recall'].items():
        if key=='v1': continue
        algo,pool=key.split('/')
        quality.append(dict(line=line,dataset=r['dataset'],dim=r['dim'],filter=r['sweep'],k=r['k'],algorithm=algo,pool=pool,recall=recall,queries=r['n_quality_queries']))
write('v3_quality.csv',quality)
x=[r for r in quality if r['dim']==128 and r['algorithm']=='v3_simhash512' and r['pool']=='linrN']
print('512 bits, 1% of catalogue, d128:',min(r['recall'] for r in x),max(r['recall'] for r in x), 'rows', len(x))
print('512 bits, 1% catalogue strict wins:',[(r['dataset'],r['filter'],r['batch'],round(r['pass_rate'],3),round(r['ratio'],3),r['unstable']) for r in vs if r['algorithm']=='v3_simhash512' and r['pool']=='linrN' and r['ratio']<1])
print('NOT:',bs)
