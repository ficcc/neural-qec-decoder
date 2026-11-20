"""E5: decoding latency. MWPM (per-shot decode, decode_batch) vs GNN (graph build + forward) at
B in {1, 64, 4096}; median and p99 per call, reported per shot and per round (÷ d).
Hardware needs ~1 µs per round; this measures how far a Python/PyTorch decoder is from that."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from experiments.common import gnn
from src.circuits import dem_of, noiseless_circuit
from src.calibrate import matching_from_dem
from src.evaluate import sample_shots
from src.graph import build_graph_batch, detector_table
from src.noise import apply_noise, si1000_model

P, BATCHES = 0.004, (1, 64, 4096)


def _timed(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return np.array(ts)


def stats(ts, shots_per_call, d):
    per_shot = ts / shots_per_call * 1e6
    return {"median_us_shot": float(np.median(per_shot)), "p99_us_shot": float(np.percentile(per_shot, 99)),
            "median_us_round": float(np.median(per_shot) / d), "p99_us_round": float(np.percentile(per_shot, 99) / d)}


def measure(d, device, shots=10_000):
    c = noiseless_circuit(d, d)
    noisy = apply_noise(c, si1000_model(c, P))
    dets, _ = sample_shots(noisy, shots, seed=7)
    m = matching_from_dem(dem_of(noisy))
    out = {"d": d, "device": device, "p": P}
    reps = min(2000, shots)
    for i in range(10):
        m.decode(dets[i])
    out["mwpm_single"] = stats(_timed(lambda it=iter(dets): m.decode(next(it)), reps), 1, d)
    out["mwpm_batch4096"] = stats(_timed(lambda: m.decode_batch(dets[:4096]), 5), 4096, d)
    model, cfg = gnn(f"e1/d{d}_s0")
    model = model.to(device).eval()
    table = detector_table(c, d)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for b in BATCHES:
        n_calls = 300 if b == 1 else (50 if b == 64 else 5)
        chunks = [dets[(k * b) % (shots - b):(k * b) % (shots - b) + b] for k in range(n_calls + 10)]
        it = iter(chunks)
        g_ts, f_ts = [], []
        with torch.no_grad():
            for k in range(n_calls + 10):
                ch = next(it)
                t = time.perf_counter(); g = build_graph_batch(ch, table, d=d, rounds=d, k=cfg.k).to(device); sync()
                t1 = time.perf_counter(); model(g); sync(); t2 = time.perf_counter()
                if k >= 10:   # warm-up excluded
                    g_ts.append(t1 - t); f_ts.append(t2 - t1)
        g_ts, f_ts = np.array(g_ts), np.array(f_ts)
        out[f"gnn_b{b}_graph"] = stats(g_ts, b, d)
        out[f"gnn_b{b}_forward"] = stats(f_ts, b, d)
        out[f"gnn_b{b}_total"] = stats(g_ts + f_ts, b, d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--d", type=int, nargs="*", default=[3, 5, 7])
    a = ap.parse_args(argv)
    meta = {"cpu": platform.processor() or platform.machine(), "torch_threads": torch.get_num_threads(),
            "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None}
    res = {"meta": meta, "rows": [measure(d, a.device) for d in a.d]}
    Path("results").mkdir(exist_ok=True)
    Path(f"results/e5_{a.device}.json").write_text(json.dumps(res, indent=1))
    for r in res["rows"]:
        print(r["d"], r["device"], {k: round(v["median_us_round"], 2) for k, v in r.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
