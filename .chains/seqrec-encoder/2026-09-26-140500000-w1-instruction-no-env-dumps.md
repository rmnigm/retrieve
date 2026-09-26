---
chain: "seqrec-encoder"
branch: "w1-encoder"
parent: "2026-09-26-135100000-w1-amendment-1-hstu-softmax-scope.md"
nextStep: "W1: continue under the rule."
created: "2026-09-26T11:05:00Z"
---

# W1 instruction: never print env files

Sent to w1-encoder and w2-etl after the user's rule:

> Hard rule from the user, effective immediately: never cat, print, grep, head or dump env/secrets files (/etc/retrieve-pod.env, /workspace/.pod-home/secrets.env, any .env) and never run env, printenv, set or declare -x dumps. Check a single non-secret variable by name only.
