# Fine-tuning a small embedding model on duplicate pairs

**In plain words:** a small search model was taught what "the same problem" looks like in IT tickets, using 2,696 pairs of duplicate questions that moderators had already linked. It was then scored on 1,079 duplicates it had never seen.

Script: `experiments/finetune/finetune_e5.py`. Data: the serverfault set built by the pipeline (`$OPS_BENCH_DIR/sf/gold`: 320,296 tickets as the search corpus; dev and held-out queries are split so no original appears in both, and the script asserts it).

Model `intfloat/e5-small-v2` (33M parameters, 384 dimensions), `MultipleNegativesRankingLoss` (every other ticket in the batch is a negative), 1 epoch = 85 steps of 32 pairs, learning rate 2e-5, bf16 activations and gradient checkpointing to fit a 4.8 GiB GPU memory cap, tickets cut at 256 tokens. Training pairs are (cleaned query text, the moderator-marked original ticket) from the dev split only; queries flagged as leaked were left out of training.

## Result (held-out, scored once, 1,079 queries, full 320,296-ticket corpus)
| | Recall@5 | Recall@10 | MRR |
|---|---|---|---|
| e5-small-v2, no training | 0.233 | 0.294 | 0.177 |
| e5-small-v2, fine-tuned | **0.276** | **0.361** | **0.215** |
| difference (paired bootstrap, 10,000 resamples, seed 1) | +0.044 [+0.023, +0.064] | +0.068 [+0.044, +0.091] | +0.039 [+0.025, +0.052] |

## Reading it
- 85 steps on 2.7k pairs give a small but real gain: the confidence intervals exclude 0. About 4 more duplicates in 100 are found in the top 5.
- It is not a large effect, and it was not compared with a bigger off-the-shelf model in this run. Do not read it as "fine-tuning beats bge-m3"; it says domain pairs help a small model.
- Training took 54 seconds on one AMD GPU (peak 1.41 GiB); scoring the held-out split against 320k tickets took about an hour per model while the GPU was shared.

## Caveats
One run, one seed, one epoch, no hyperparameter search. The held-out split was scored once and dev was not used for tuning either (no early stopping). "Original found" means any original of the duplicate is in the top k. Per-query rows for the base model were not saved in this run (the script now saves them); the tuned per-query rows are in `experiments/finetune/tuned-rows.json`.

Reproduce (from the repository root, after the core pipeline has produced `$OPS_BENCH_DIR/sf/gold`):

    python -I experiments/finetune/finetune_e5.py --out /tmp/finetune.json

Data: Stack Exchange data dump, CC BY-SA 4.0; attribute the Stack Exchange contributors (see DATA.md).
