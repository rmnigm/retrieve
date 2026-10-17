import sys, yaml
from pathlib import Path
sys.path[:0]=['/scratch/wt/export-r4/evaluation','/scratch/wt/export-r4/retrieve/src']
from bench.config import load_matrix
S=Path('/scratch/wt/export-r4/results/final-run-config/suites.yaml')
C=Path('/scratch/wt/export-r4/evaluation/config')
suites=yaml.safe_load(S.read_text())
for name,s in suites.items():
    if not isinstance(s,dict) or 'datasets' not in s: continue
    errs=set(); ok=0
    for ds in s['datasets']:
        try:
            load_matrix(C/f'{ds}.yaml',S,name); ok+=1
        except Exception as e:
            errs.add(type(e).__name__+': '+str(e).split(':')[-1].strip()[:90])
    print(f"{name}: {'OK' if not errs else 'FAIL'} ({ok}/{len(s['datasets'])})", '; '.join(sorted(errs)))
