import os
import threading
import time
from queue import Queue, Empty

import torch
import waitress
from pyroaring import BitMap
from flask import Flask, request, Response
import msgpack
from model import load_model, build_model_from_checkpoint, policy_width, autocast, DEVICE, GLOBAL_MAX, ACTIONS_MAX
from vocab import FeatureVocab

# Threading config
TORCH_THREADS = 1 #max(1, os.cpu_count() // 2)
torch.set_num_threads(TORCH_THREADS)

# Batching config: after the first queued request, wait up to MAX_WAIT_MS for more so several
# game threads share one forward pass (per-call overhead dominates on small batches).
MAX_BATCH = int(os.environ.get("MZ_MAX_BATCH", 16))
MAX_WAIT_MS = float(os.environ.get("MZ_BATCH_WAIT_MS", 0))

#module state
server_model = None
IGNORE_BM = None
VALID_RANGE = None
VOCAB = None  # FeatureVocab for dense-vocab checkpoints; None for full-table ones

app = Flask(__name__)

req_counter = 0
req_counter_lock = threading.Lock()

STATS = {"reqs": 0, "bags": 0, "batches": 0, "lat_ms": [], "value_sum": 0.0}
STATS_LOCK = threading.Lock()
STATS_EVERY_S = 30


def stats_loop():
    """Print one aggregate line per interval instead of a line per request."""
    while True:
        time.sleep(STATS_EVERY_S)
        with STATS_LOCK:
            snap = dict(STATS); lat = sorted(STATS["lat_ms"])
            STATS.update(reqs=0, bags=0, batches=0, lat_ms=[], value_sum=0.0)
        if not snap["reqs"]:
            continue
        p50 = lat[len(lat) // 2]; p95 = lat[int(len(lat) * 0.95)]
        print(f"[STATS] window_s={STATS_EVERY_S} reqs={snap['reqs']} bags={snap['bags']} "
              f"avg_batch={snap['bags'] / max(snap['batches'], 1):.2f} p50_ms={p50:.1f} p95_ms={p95:.1f} "
              f"value_mean={snap['value_sum'] / max(snap['bags'], 1):.3f}", flush=True)


def init(deck: str, version: int, port: int, checkpoint: str | None = None):
    global server_model, IGNORE_BM, VALID_RANGE, VOCAB

    model_dir = f"models/{deck}/ver{version}"
    ignore_path = f"{model_dir}/ignore.roar"
    model_path = f"{model_dir}/model.pt.gz"
    if checkpoint:  # e.g. "gen2" -> frozen snapshot written by train.py
        ignore_path = f"{model_dir}/{checkpoint}.ignore.roar"
        model_path = f"{model_dir}/{checkpoint}.pt.gz"

    ckpt = load_model(model_path)
    if "feature_vocab" in ckpt:
        # dense vocab: ids are mapped to rows; ids outside the vocab are the ignored ones
        VOCAB = FeatureVocab.from_state_dict(ckpt["feature_vocab"])
        # rows only mean anything under the encoding that built them
        VOCAB.require_encoding(GLOBAL_MAX)
    else:
        with open(ignore_path, "rb") as f:
            IGNORE_BM = BitMap.deserialize(f.read())
        VALID_RANGE = BitMap(range(GLOBAL_MAX))
    server_model = build_model_from_checkpoint(ckpt).to(DEVICE).eval()
    head = policy_width(server_model)
    if head != ACTIONS_MAX:
        # the JVM indexes these logits with its own ActionEncoder; a width mismatch would
        # silently point priors at the wrong actions
        raise SystemExit(f"model {model_path} has {head}-wide policy heads but this run uses "
                         f"{ACTIONS_MAX} (MZ_ACTION_VOCAB={os.environ.get('MZ_ACTION_VOCAB')})")
    if os.environ.get("MZ_INFER_PAD_BUCKET"):
        # inference sees one shape per request anyway; finer buckets waste less compute on padding
        import model as _model
        _model.PAD_BUCKET = int(os.environ["MZ_INFER_PAD_BUCKET"])
    if os.environ.get("MZ_INFER_DTYPE") == "float16":
        # inference-only half precision: same weights, ~half the GPU work (see experiments/fdn/README)
        server_model = server_model.half()

    threading.Thread(target=worker_loop, daemon=True).start()
    threading.Thread(target=stats_loop, daemon=True).start()

    print(f"[INIT] deck={deck} ver={version} model={model_path} port={port} device={DEVICE} action_dim={head}", flush=True)
    waitress.serve(app, host="127.0.0.1", port=port, threads=6)

class Pending:
    __slots__ = ("idx", "off", "evt", "out", "req_id", "pre_count", "post_count", "t_recv", "t_done", "num_bags")

    def __init__(self, req_id, indices, offsets):
        self.req_id = req_id
        self.pre_count = len(indices)
        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.t_done = 0.0

        indices, offsets, num_bags = apply_ignore(indices, offsets)
        self.post_count = len(indices)
        self.num_bags = num_bags

        self.idx = torch.tensor(indices, dtype=torch.long)
        self.off = torch.tensor(offsets, dtype=torch.long)


def apply_ignore(indices: list[int], offsets: list[int] | None):
    if not offsets:
        offsets = [0]

    if VOCAB is not None:
        rows, new_offsets = VOCAB.map_bags(indices, offsets)
        return rows.tolist(), new_offsets.tolist(), len(new_offsets)

    if len(offsets) == 1:
        # Single bag - pure bitmap ops in C
        kept_bm = (BitMap(indices) - IGNORE_BM) & VALID_RANGE
        return list(kept_bm), [0], 1

    # Multi-bag
    n = len(indices)
    new_indices = []
    new_offsets = [0]

    for b in range(len(offsets)):
        start = offsets[b]
        end = offsets[b + 1] if b + 1 < len(offsets) else n

        kept_bm = (BitMap(indices[start:end]) - IGNORE_BM) & VALID_RANGE
        new_indices.extend(kept_bm)
        new_offsets.append(len(new_indices))

    new_offsets = new_offsets[:-1]
    return new_indices, new_offsets, len(new_offsets)


Q: "Queue[Pending]" = Queue(maxsize=4096)


def worker_loop():
    while True:
        p0 = Q.get()
        batch = [p0]

        # Collect more requests up to MAX_BATCH or MAX_WAIT_MS
        deadline = time.perf_counter() + (MAX_WAIT_MS / 1000.0)
        while len(batch) < MAX_BATCH or not Q.empty():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                remaining = 0
                if Q.empty():
                    break
            try:
                batch.append(Q.get(timeout=remaining))
            except Empty:
                break

        bag_counts = [p.num_bags for p in batch]

        # Fast path: single request
        if len(batch) == 1:
            idx = batch[0].idx.to(DEVICE, non_blocking=True)
            off = batch[0].off.to(DEVICE, non_blocking=True)
        else:
            # Concatenate indices
            idx = torch.cat([p.idx for p in batch]).to(DEVICE, non_blocking=True)

            # Vectorized offset adjustment
            all_off = torch.cat([p.off for p in batch])
            idx_lens = torch.tensor([len(p.idx) for p in batch])
            bag_counts_t = torch.tensor(bag_counts)
            adjustments = torch.repeat_interleave(
                torch.cat([torch.tensor([0]), idx_lens.cumsum(0)[:-1]]),
                bag_counts_t
            )
            off = (all_off + adjustments).to(DEVICE, non_blocking=True)

        if VOCAB is None:
            idx = idx % GLOBAL_MAX   # raw feature ids; dense-vocab rows are already < len(VOCAB)
        # Single forward pass. autocast() comes from model.py and picks the right device:
        # torch.amp.autocast('cuda') would break every CPU/MPS worker.
        with torch.no_grad(), autocast():
            pA, pB, tgt, bin2, val = server_model(idx, off)

        # Move to CPU once
        pA = pA.cpu()
        pB = pB.cpu()
        tgt = tgt.cpu()
        bin2 = bin2.cpu()
        val = val.cpu()

        # Split results back to individual requests
        row = 0
        for p, num_bags in zip(batch, bag_counts):
            if num_bags == 1:
                p.out = {
                    "policy_player": pA[row].tolist(),
                    "policy_opponent": pB[row].tolist(),
                    "policy_target": tgt[row].tolist(),
                    "policy_binary": bin2[row].tolist(),
                    "value": float(val[row].item()),
                }
            else:
                p.out = [
                    {
                        "policy_player": pA[row + i].tolist(),
                        "policy_opponent": pB[row + i].tolist(),
                        "policy_target": tgt[row + i].tolist(),
                        "policy_binary": bin2[row + i].tolist(),
                        "value": float(val[row + i].item()),
                    }
                    for i in range(num_bags)
                ]
            row += num_bags
            p.t_done = time.perf_counter()
            p.evt.set()

        with STATS_LOCK:
            STATS["batches"] += 1
            STATS["bags"] += row
            STATS["value_sum"] += float(val[:row].float().sum())


#threading.Thread(target=worker_loop, daemon=True).start()


@app.post("/evaluate")
def evaluate():
    global req_counter

    data = msgpack.unpackb(request.data, raw=False)
    with req_counter_lock:
        req_counter += 1

    indices = data.get("indices", [])
    offsets = data.get("offsets", [])
    pending = Pending(req_counter, indices, offsets)

    Q.put(pending)
    pending.evt.wait()

    with STATS_LOCK:
        STATS["reqs"] += 1
        STATS["lat_ms"].append((pending.t_done - pending.t_recv) * 1000.0)

    return Response(msgpack.packb(pending.out, use_bin_type=True), mimetype="application/x-msgpack")


@app.get("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    import argparse
    import waitress
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--port", type=int, default=50052)
    parser.add_argument("--checkpoint", default=None, help="frozen snapshot name, e.g. gen2")
    args = parser.parse_args()
    init(args.deck, args.version, args.port, args.checkpoint)