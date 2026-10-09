# YFCC int8-precision mechanism check

Controller 2026-10-10 (from exhibits): is SilverTorch's recall ceiling on YFCC 10 M d192 the single global int8 item scale? Quality
only, campaign-v2.5. Driver [`../../campaign-v2/v-pod1-run/yfcc-int8.sh`](../../campaign-v2/v-pod1-run/yfcc-int8.sh); Hub
`artifacts/yfcc-int8`.

| file | what |
|---|---|
| [`int8_check.py`](int8_check.py) | one SilverTorch triton index (n_lists 4096, n_probe 256), its own probed clusters per query; recall_oracle@100 with the shipped module, the same arithmetic recomputed densely (method check), per-row int8 item scales on the int32 path, and fp16 scoring |
