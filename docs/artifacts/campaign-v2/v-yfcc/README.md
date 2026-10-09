# V-YFCC: yfcc10m-synth `synth` in chunks (pods 1 and c)

Roadmap V-YFCC, at campaign-v2.1. The synth suite's 285 cells are split into 40 chunks, each taken by
whichever pod frees first; claims live on `.chains/v-yfcc/`. NOT CITABLE.

| file | what |
|---|---|
| [`yfcc_chunks.py`](yfcc_chunks.py) | the chunk plan, checked on CPU: per arm group (V3; V1 + V2; postfilter; SilverTorch triton clause; SilverTorch triton + official bloom) × filter kind × sweep, all seeds; every chunk a CLI narrowing whose resume keys and interleave units equal the full expansion's; together exactly the 285 cells |
| [`synth-chunks.sh`](synth-chunks.sh) | pod 1's driver (on [`../v-pod1-run/common.sh`](../v-pod1-run/common.sh)): oracles, then for each chunk not named in a `.chains/v-yfcc/` note: a claim note, `bench run --resume --interleave` into `/scratch/v-yfcc-synth/<chunk>`, `bench upload --verify` to `artifacts/v-yfcc-synth-<chunk>`, a done note; stops between chunks on `/scratch/v21/v-yfcc-synth.stop` |

Claim notes carry `chunk: <name>` and `state: claim | done | failed` lines; a chunk is free only when no note names it.
