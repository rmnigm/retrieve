# k64-refit prediction (written 2026-09-26 22:13 UTC, before launch)

R-k64 measured: 249 s/epoch train (2,193 batches), saves ~100 s/epoch, peak 62.6 GB. train_on_val adds
24,503 rows (+4.4 %), no val eval.
- **Epoch ~260 s train + ~100 s saves; 4 epochs ~24 min**, plus the final saves and test eval (~2 min).
- Peak ~63 GB.
- Quality guess (not a measurement): test ndcg@10 between R-k64's 0.0046 and the val-day popularity
  list's 0.0314; the val day holds 47.6 % of the test targets, 13 % of them only there.
