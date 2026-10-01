| metric | ours (Gemma-4 12B, local) | Strands Decider 2B v19 (local) | TypeSafe Jev API (hosted) |
|---|---|---|---|
| noul acc (/15) | 15 | 15 | 15 |
| noul Brier (lower better) | 0.0 | 0.03379 | 0.00035 |
| choice acc (/12) | 11 | 12 | 12 |
| choice mean P(correct) | 0.924 | 0.964 | 0.993 |
| score round-acc (/12) | 11 | 5 | 12 |
| score pairwise ordering | 1.0 | 0.958 | 1.0 |
| score MAE (levels) | 0.09 | 0.483 | 0.004 |
| load s | 14.8 | 45.8 | 0.0 |
| 1-q p50 ms | 117 | 96 | 144 |
| 1-q p95 ms | 473 | 190 | 207 |
| 3-q p50 ms | 332 | 121 | 164 |
