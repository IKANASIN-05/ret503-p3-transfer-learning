"""Ukur latensi inferensi 1 citra 224x224: MobileNetV3-Small vs ResNet-18.
Jalankan di perangkat target (mis. Raspberry Pi) untuk angka yang bermakna."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from torchvision import models

ap = argparse.ArgumentParser(); ap.add_argument("--runs", type=int, default=100)
ap.add_argument("--threads", type=int, default=0, help="0 = default PyTorch"); a = ap.parse_args()
if a.threads: torch.set_num_threads(a.threads)

out = {}
for name, ctor in [("mobilenet_v3_small", models.mobilenet_v3_small), ("resnet18", models.resnet18)]:
    m = ctor(weights=None).eval(); x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        for _ in range(20): m(x)                      # warm-up
        t = []
        for _ in range(a.runs):
            s = time.perf_counter(); m(x); t.append((time.perf_counter() - s) * 1000)
    out[name] = dict(mean_ms=float(np.mean(t)), p50_ms=float(np.percentile(t, 50)),
                     p95_ms=float(np.percentile(t, 95)), fps=float(1000 / np.mean(t)))
    print(f"{name:20s} mean {out[name]['mean_ms']:.1f} ms | p95 {out[name]['p95_ms']:.1f} ms | {out[name]['fps']:.1f} FPS")
print("Anggaran inferensi pada slide 16: 35 ms")
Path("results").mkdir(exist_ok=True); Path("results/latency.json").write_text(json.dumps(out, indent=2))
