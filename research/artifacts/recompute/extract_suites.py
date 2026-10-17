import re, sys
D='/scratch/recompute/cfg/'
want=[('main.yaml',['filter','c3','c3-real','v3bits']),('fr.yaml',['fr-curve-d128','fr-linr-bs64','fr-linr-bs256','fr-yambda']),('lite.yaml',['fr-timed-clause-lite-d128','fr-timed-clause-lite-d256','fr-timed-clause-lite-d768'])]
out=[]
for f,keys in want:
    lines=open(D+f).read().split('\n')
    tops=[i for i,l in enumerate(lines) if re.match(r'^[A-Za-z0-9_-]+:',l)]+[len(lines)]
    for k in keys:
        i=next(i for i in tops if re.match(rf'^{re.escape(k)}:',lines[i]))
        j=tops[tops.index(i)+1]
        blk=[]
        for l in lines[i:j]:
            l=re.sub(r'\s*#.*$','',l)
            if l.strip()=='' : continue
            l=l.replace('official-fork','cuda')
            l=re.sub(r'\bofficial\b','published',l)
            blk.append(l)
        blk[0]=blk[0].replace('fr-','',1)
        out.append('\n'.join(blk))
print('\n'.join(out))
