"""Fine-tune a small embedding model on ops-ticket duplicate pairs and score it on the held-out split, once.

Inputs : a gold directory with tickets.parquet (id, ticket_text), gold_dup_dev.jsonl and gold_dup_heldout.jsonl.
Outputs: --out JSON (base vs tuned metrics, paired bootstrap, loss log), plus <out>.tuned-model/,
         <out>.tuned-rows.json and <out>.base-rows.json (per-query scores).

Usage: python finetune_e5.py [--gold DIR] [--model intfloat/e5-small-v2] [--epochs 1] [--bs 32] [--out results.json]
Train pairs  = (query_clean, text of its moderator-marked original) from the DEV split only. Loss = MultipleNegativesRankingLoss (in-batch negatives).
Eval        = HELD-OUT split, recall@5/10 and MRR over the full ticket corpus, base model vs fine-tuned, paired bootstrap over queries.
Guards      = no held-out original may appear among the training positives (checked); VRAM capped; the held-out split is touched once, after training."""
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch
from sentence_transformers import SentenceTransformer, losses

ap = argparse.ArgumentParser()
ap.add_argument("--gold", default=os.path.join(os.environ.get("OPS_BENCH_DIR", os.path.expanduser("~/ops-benchmark")), "sf/gold"))
ap.add_argument("--model", default="intfloat/e5-small-v2")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--vram", type=float, default=0.3)
ap.add_argument("--max-seq", type=int, default=256)
ap.add_argument("--out", default="results.json")
ap.add_argument("--seed", type=int, default=13)
a = ap.parse_args()

# Cap this process's share of the (shared) GPU and fix the seeds.
torch.cuda.set_per_process_memory_fraction(a.vram)
random.seed(a.seed)
torch.manual_seed(a.seed)

G = Path(a.gold)

# Ticket id -> row position in the parquet, so gold originals can be mapped to corpus rows.
ids = [str(i) for i in pq.read_table(G / "tickets.parquet", columns=["id"]).column("id").to_pylist()]
row = {i: k for k, i in enumerate(ids)}


def stream(bs=20000):  # corpus is streamed: never all ticket texts in memory at once
    for b in pq.ParquetFile(G / "tickets.parquet").iter_batches(batch_size=bs, columns=["ticket_text"]):
        yield b.column(0).to_pylist()  # fresh handle each pass (reusing one broke the 2nd pass)


load = lambda n: [json.loads(l) for l in open(G / n, encoding="utf-8")]


def origs(g):
    # "originals" is sometimes a JSON string, sometimes already a list.
    return [str(o) for o in (json.loads(g["originals"]) if isinstance(g["originals"], str) else g["originals"])]


dev, held = load("gold_dup_dev.jsonl"), load("gold_dup_heldout.jsonl")
dev = [g for g in dev if not (g.get("leaked") in (True, "True"))]  # train on cleaned queries only

# Pull only the texts of the originals we need out of the streamed corpus.
need = {row[o] for g in dev for o in origs(g) if o in row}
text_of = {}
k0 = 0
for chunk in stream():
    for j, t in enumerate(chunk):
        if k0 + j in need:
            text_of[k0 + j] = t
    k0 += len(chunk)

# Training pairs: (cleaned query, text of its moderator-marked original).
pairs = [(g["query_clean"], text_of[row[o]]) for g in dev for o in origs(g) if o in row]

# The held-out split must stay unseen: no held-out original may be a training positive.
train_pos = {o for g in dev for o in origs(g)}
held_pos = {o for g in held for o in origs(g)}
assert not (train_pos & held_pos), "held-out originals overlap training positives"
print(f"{len(pairs)} training pairs from {len(dev)} dev queries; corpus {len(ids)}; held-out {len(held)}", flush=True)


def encode_corpus(m, bs=64):
    parts = []
    for chunk in stream():
        # Sort by length so each batch has little padding; the original order is restored below.
        order = np.argsort([len(t) for t in chunk])
        # e5 models were trained with the "passage: " prefix; vectors are normalized so dot product = cosine.
        e = m.encode(
            ["passage: " + chunk[i] for i in order],
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype(np.float16)
        out = np.empty_like(e)
        out[order] = e
        parts.append(out)
    return np.concatenate(parts)


def evaluate(m):
    C = torch.from_numpy(encode_corpus(m)).cuda()
    res = []
    for g in held:
        # "query: " is the matching e5 query prefix.
        q = torch.from_numpy(m.encode(["query: " + g["query_clean"]], normalize_embeddings=True)).cuda().half()
        top = torch.topk((C @ q.T).squeeze(1).float(), 100).indices.cpu().numpy()
        rel = {row[o] for o in origs(g) if o in row}

        # Rank (1-based) of the first relevant original, or None if it is not in the top 100.
        rk = next((p + 1 for p, i in enumerate(top) if i in rel), None)
        res.append({
            "qid": g["qid"],
            "r5": float(rk is not None and rk <= 5),
            "r10": float(rk is not None and rk <= 10),
            "mrr": 1 / rk if rk else 0.0,
        })
    del C
    torch.cuda.empty_cache()
    return res


m = SentenceTransformer(a.model, device="cuda")
m.max_seq_length = a.max_seq  # tickets are cut at 256 tokens by default, to match the baseline arm
loss = losses.MultipleNegativesRankingLoss(m)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr)

# Linear warmup over the first 10% of steps, then linear decay to zero.
steps = math.ceil(len(pairs) / a.bs) * a.epochs
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: min(1, (s + 1) / max(1, 0.1 * steps)) * max(0.0, 1 - s / steps),
)
base_dir = Path(a.out).with_suffix(".base.json")
t0 = time.time()
step = 0
log = []
m[0].auto_model.gradient_checkpointing_enable()  # trade compute for memory: the 4.8 GiB VRAM budget
m.train()
for ep in range(a.epochs):
    random.shuffle(pairs)
    for b in range(0, len(pairs) - a.bs + 1, a.bs):
        batch = pairs[b:b + a.bs]
        # Same e5 prefixes as at evaluation time, so training matches how the model is queried.
        fq = m.tokenize(["query: " + q for q, _ in batch])
        fd = m.tokenize(["passage: " + d for _, d in batch])
        fq = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in fq.items()}
        fd = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in fd.items()}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = loss([fq, fd], None)  # bf16 activations: fp32 at batch 32 x 256 tokens exceeded the 4.8 GiB cap

        l.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        if step % 20 == 0:
            log.append((step, float(l)))
            print(f"step {step}/{steps} loss {float(l):.3f}", flush=True)

train_s = time.time() - t0
peak = torch.cuda.max_memory_allocated() / 2**30

# The held-out split is scored here, once per model (tuned, then base).
m.eval()
m.save(str(Path(a.out).with_suffix('.tuned-model')))
tuned = evaluate(m)
json.dump(tuned, open(Path(a.out).with_suffix('.tuned-rows.json'), 'w'))  # checkpoint: do not lose 15 min of scoring to a later failure

base_m = SentenceTransformer(a.model, device="cuda")
base_m.max_seq_length = a.max_seq
base = evaluate(base_m)
json.dump(base, open(Path(a.out).with_suffix('.base-rows.json'), 'w'))

# Paired bootstrap over queries: resample the per-query tuned-minus-base differences.
rng = np.random.RandomState(1)
n = len(held)


def ci(k):
    d = np.array([t[k] - b[k] for t, b in zip(tuned, base)])
    mm = d[rng.randint(0, n, (10000, n))].mean(1)
    return {"diff": float(d.mean()), "ci95": [float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]}


out = {
    "model": a.model, "gold": str(G), "train_pairs": len(pairs), "epochs": a.epochs, "batch": a.bs, "lr": a.lr,
    "steps": steps, "train_seconds": round(train_s), "vram_peak_gib_training": round(peak, 2),
    "heldout_n": n,
    "base": {k: float(np.mean([r[k] for r in base])) for k in ("r5", "r10", "mrr")},
    "tuned": {k: float(np.mean([r[k] for r in tuned])) for k in ("r5", "r10", "mrr")},
    "paired_bootstrap": {k: ci(k) for k in ("r5", "r10", "mrr")},
    "loss_log": log,
}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps({k: out[k] for k in ("base", "tuned", "paired_bootstrap")}, indent=1))
