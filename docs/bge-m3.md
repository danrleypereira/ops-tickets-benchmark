# bge-m3: the strongest model here, best with the votes prior (adds `f3_embed_bge.py`, `f14_bge_prior.py`)
**What it is.** BAAI/bge-m3 (1,024 numbers per ticket, no prefix, fp16, full ticket, 256 tokens) instead of e5-base, and the votes prior of the votes-prior approach applied to its vectors (`alpha` 0.3, tuned on the same 1,000 development queries, held-out run once).

**Result** (serverfault, 1,079 held-out questions, clean queries):
- Recall@10 31.8 out of 100 (95% interval 28.9 to 34.8) against 30.9 for e5-base. Paired difference in Recall@10 +0.9 points (-1.1 to +3.0): not significant. The right ticket is first more often: Recall@1 15.7 against 13.4, paired difference +2.2 points (+0.7 to +3.8), significant.
- With the votes prior: Recall@10 45.8 (paired difference to bge-m3 alone +14.0 points, +11.8 to +16.3). That is 5.0 points above BM25 with the prior (+1.9 to +8.3) and 7.9 points above e5 with the prior (+5.5 to +10.3).
- Cost: embedding all 320,296 tickets took 3,298 s (about 55 minutes, 88 tickets/s) against 1,466 s for e5-base (that e5 run was not timed on an exclusive GPU, so the ratio of about 2.2 is approximate).

**How to run.** `python -I pipeline/f3_embed_bge.py $SF/gold $SF/emb_bgem3 256 0.35 32`, then `python -I pipeline/f4_eval.py $SF/gold $SF/emb_bgem3 $SF/results/bgem3_heldout.json heldout`, then (needs `lxml` and `Votes.xml`) `python -I pipeline/f14_bge_prior.py`.

**Limits.** One run per arm; the prior is tuned on a development slice.
