# Votes prior: the biggest lift (adds `priors/`)
**What it is.** Moderators close a new question against the established, popular older question, so popular tickets are more likely to be the right answer. The prior adds `alpha * log(1 + net votes)` to each candidate's similarity (rescaled to 0..1 inside the top 100). Only votes cast **before the query's day** count (`votes_asof.py` rebuilds the history from `Votes.xml`), so nothing from the future leaks in. `alpha` is chosen on 1,000 development queries (seed 21) and applied once to the held-out set.

**Result** (serverfault, 1,079 held-out questions, clean queries, Recall@10, out of 100):

| Method | Without the prior | With the prior | Paired difference (95% interval) |
|---|---|---|---|
| BM25 | 22 | 41 | +18.6 points (+16.0 to +21.2) |
| e5-base, full ticket, 256 tokens | 31 | 38 | +7.0 points (+5.4 to +8.8) |

**How to run.** `priors/README.md` lists the commands (set `DATA_DIR=$OPS_BENCH_DIR/sf`).

**Limits.** The signal fits how duplicates are closed (canonical questions are popular) and may not transfer unchanged to another ticket system; `alpha` was tuned on a development slice; one run.
