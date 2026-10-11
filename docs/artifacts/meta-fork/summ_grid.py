import json,glob,sys
d=sys.argv[1]; n=0;xbad=[];loss=[];rec=[]
for f in sorted(glob.glob(d+'/*.json')):
    for r in json.load(open(f)):
        n+=1
        if r.get('fork_bloom')=='meta' or r['mode']!='bloom':
            if not r['vs_upstream']['equal']: xbad.append((f.split('/')[-1][:60],r['bs'],r['k']))
        if 'recall_fork' in r: rec.append((r['dataset'],r['n_probe'],r['bs'],r['k'],r['recall_fork'],r['false_positives_fork']))
        for m in ('eager','graph'):
            v=r.get(f'{m}_fork_over_triton')
            if v and v>1.0: loss.append((round(v,3),m,r['dataset'][:6],r['dim'],r['mode'],r['sweep'],r.get('bloom_path'),r.get('fork_bloom'),r['n_probe'],r['bs'],r['k']))
print('cells',n,'X fails',xbad)
print('recall<1 or fp>0:',[x for x in rec if x[4]<1 or x[5]>0], 'of', len(rec))
for l in sorted(loss,reverse=True): print(l)
