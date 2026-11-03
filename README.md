# Ops-ticket search benchmark

Does an earlier ticket already solve this one? A reproducible benchmark of search methods on public support-style tickets:
[serverfault.com](https://serverfault.com) (320,296 tickets in the search base). The answer key is free: moderators already linked
new questions to the older question that answers them (4,790 duplicate pairs). Data: Stack Exchange data dump, CC BY-SA 4.0
(see [DATA.md](DATA.md)). This is a measurement project, not a production system.

## Results (the core)
Held-out set: 1,079 serverfault questions that were closed as duplicates of an older question, clean queries (the "Possible
duplicate" banner and links removed). "Out of 100" = Recall@10: in how many of 100 questions the right older ticket is among the
10 returned. Intervals are 95% paired bootstrap intervals over the same questions. Runs from 2026-11-02 to 2026-11-03.

| Method | Right ticket in the top 10, out of 100 |
|---|---|
| BM25 keywords (full ticket) | 22 (95% interval 20 to 25) |
| e5-base meaning vectors, full ticket, 256 tokens | 31 (28 to 34); paired difference to BM25 +8.7 points (+5.7 to +11.7) |
| e5-base, title + question only, 512 tokens | 30 (27 to 33); paired difference to the 256-token run -0.8 points (-2.7 to +1.0): no measurable difference |
| e5-base, full ticket, 512 tokens | 27 (24 to 30); paired difference to the 256-token run -3.9 points (-5.7 to -2.0): worse; the reason was not tested |
| BM25 + e5 fused (reciprocal rank fusion, k = 60) | 28 (26 to 31); not better than e5 alone |

Leaving the "Possible duplicate" banner in the query inflates Recall@10 by about 5.6 points (BM25) and 5.8 points (e5). On one AMD
GPU (ROCm) embedding all 320,296 tickets took 2,182 s (title + question, 512 tokens, exclusive GPU) and 3,280 s (full ticket, 512
tokens, exclusive GPU); the same title + question job took 7,159 s while the GPU was shared. The batch size differed between those
runs (16 and 64), so the time gap is not only the token count. A difference whose interval contains 0 is not claimed.

More approaches (a votes signal, bge-m3, pgvector HNSW, tenant isolation, ...) arrive as separate pull requests, each with a
`docs/<approach>.md` that states what was measured; the ones that did not help stay open as documented experiments.

## Layout
| Folder | What is in it |
|---|---|
| `pipeline/` | `f0_parse` XML to Parquet, `f1_gold` tickets and the duplicate test set, `f2_gold_to_parquet`, `f5_dq` data-quality rules and `dq_show` (a readable report), `fix_tags`, `f3_embed` vectors on the GPU, `f4_eval` BM25 / dense / fusion scores with confidence intervals, `f17_paired_arms` paired differences. |
| `scripts/` | `download_data.sh` fetches the public dump. |

## Configuration
Everything lives under one folder, set with `OPS_BENCH_DIR` (default `~/ops-benchmark`):

    $OPS_BENCH_DIR/sf/    serverfault: raw/ xml/ parquet/ gold/ emb_*/ results/

Run the scripts from the repository root with `python -I pipeline/<script>.py ...` (isolated mode, so no stray module on the
path is imported).

## Reproduce, step by step
Hardware used: one AMD GPU with 16 GB (ROCm). The scripts use plain PyTorch calls that exist on CUDA builds too (not tested on an
NVIDIA card here). Install PyTorch for your hardware first, then `pip install -r requirements.txt`.

    export OPS_BENCH_DIR=$HOME/ops-benchmark
    SF=$OPS_BENCH_DIR/sf
    bash scripts/download_data.sh serverfault                       # downloads and unpacks the dump
    python -I pipeline/f0_parse.py $SF/xml $SF/parquet               # XML to Parquet
    python -I pipeline/f1_gold.py $SF/parquet $SF/gold               # tickets + duplicate test set
    python -I pipeline/f5_dq.py $SF/gold/tickets.parquet /tmp/dq.json 320296   # data-quality rules (or: python -I pipeline/dq_show.py)
    python -I pipeline/f4_eval.py $SF/gold none $SF/results/bm25_heldout.json heldout    # BM25 only
    F3_BS=64 python -I pipeline/f3_embed.py $SF/gold $SF/emb_e5base intfloat/multilingual-e5-base "query: " 256 0.3   # GPU
    python -I pipeline/f4_eval.py $SF/gold $SF/emb_e5base $SF/results/e5full256_heldout.json heldout
    python -I pipeline/f17_paired_arms.py $SF/gold/gold_dup_heldout.jsonl \
        $SF/results/e5full256_heldout_outputs $SF/results/<other arm>_outputs /tmp/paired.json

`f3_embed.py` takes `F3_FIELD=title_question` and a longer window (for example `... "query: " 512 0.15`) for the other e5 arms. The
last argument is the share of GPU memory the job may use. Each script's docstring lists its inputs, outputs and memory needs.

## Licence
Code: MIT ([LICENSE](LICENSE)). Data: CC BY-SA 4.0, not included ([DATA.md](DATA.md)).
