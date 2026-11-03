# Following related-ticket links for two-hop questions (adds `f7_graph.py`, `f7_bm25.py`, `data/`) - kept open, not merged
**What it is.** Questions that need two tickets at once. The moderators' "related" links (`PostLinks` type 1; duplicate links are never used) serve as graph edges: take the top-5 hits as seeds, add their linked tickets, and re-rank. Version 1 gave only neighbours a bonus and pushed an exact single-hop answer out of the top 10 in a control; version 2 gives a seed a bonus from its own similarity.

**Result** (hand-written two-hop questions; path recall@5 = mean share of the 2 needed tickets in the top 5; **n is 6 and 7, no interval is meaningful**):
- First held-out set (6 questions): dense 0.583, version 1 0.667; one control question (single-fact) fell from 1.0 to 0.5 with version 1.
- Fresh held-out set (7 questions, run once): dense 0.786 (4 full paths), version 1 1.000 (7), version 2 0.857 (5); version 2 kept both controls at 1.0.
- Baselines without links: BM25 is weakest (0.25 and 0.29); equal-weight fusion does not beat dense.

**Why it is not merged.** Small sample, directional only, and version 1 broke a control. The gain needs a larger question set to count.
