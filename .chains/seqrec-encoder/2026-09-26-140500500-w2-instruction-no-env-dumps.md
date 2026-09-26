---
chain: "seqrec-encoder"
branch: "w2-etl"
parent: "2026-09-26-134930500-w2-brief-etl-timestamps-prep.md"
nextStep: "W2: continue under the rule."
created: "2026-09-26T11:05:00Z"
---

# W2 instruction: never print env files

> Hard rule from the user, effective immediately: never cat, print, grep, head or dump env/secrets files (/etc/retrieve-pod.env, /workspace/.pod-home/secrets.env, any .env) and never run env, printenv, set or declare -x dumps. Check a single non-secret variable by name only.
