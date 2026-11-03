# Tag filter and "no confident match" signals (adds `f9_filter_abstain.py`, `f13_abstain_signals.py`) - kept open, not merged
**What it is.** (1) A hard tag filter: keep only candidates that share at least one tag with the new ticket. (2) Abstention: can a score tell "a match exists" from "none exists"? Tested with the original masked out of the base as the "unanswerable" case.

**Result** (1,079 held-out clean queries, e5 full ticket 256 tokens):
- The filter did not help: Recall@10 30.9 unfiltered against 29.6 filtered, paired difference -1.3 points (-2.8 to +0.2); the true original survives the filter in only 81.8% of queries (it removes the right answer in about 1 in 5).
- Four score-based confidence signals all sit near chance (AUROC, 0.5 = chance): top-1 cosine 0.545, gap between hit 1 and 2 0.539, gap between hit 1 and the mean of hits 2-10 0.547, z-score of the top-1 0.544. At the development threshold 31.7% of answerable queries were answered and 74.4% of unanswerable ones abstained. Why: with Recall@1 of 13.4, the top hit is not the original in about 87% of "answerable" queries anyway.
- A first run of the z-score signal showed AUROC 1.0: that was a bug in the masking (a large negative placeholder distorted the mean and spread); fixed, rerun gives 0.544. A perfect score is a bug until proven otherwise.

**Why it is not merged.** No gain and a negative filter result. Not tried: a reranker score or a trained classifier for the escalation decision.
