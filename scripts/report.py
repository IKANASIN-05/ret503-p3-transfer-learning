"""Buat results/tabel_hasil.md dan results/akurasi_per_epoch.png dari results/*.json"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

modes = [m for m in ["feature", "partial", "scratch"] if Path(f"results/{m}.json").exists()]
R = {m: json.loads(Path(f"results/{m}.json").read_text()) for m in modes}
rows = ["| Mode | Akurasi val terbaik | Waktu latih (detik) | Epoch pertama val >= 90% |", "|---|---|---|---|"]
for m, r in R.items():
    rows.append(f"| {m} | {r['best_val_acc']*100:.1f}% | {r['train_time_s']} | {r['epoch_acc90'] or 'tidak tercapai'} |")
Path("results/tabel_hasil.md").write_text("\n".join(rows) + "\n"); print("\n".join(rows))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for m, r in R.items():
    e = range(1, len(r["history"]["val_acc"]) + 1)
    ax[0].plot(e, r["history"]["train_acc"], label=m); ax[1].plot(e, r["history"]["val_acc"], label=m)
for a, t in zip(ax, ["Akurasi train", "Akurasi validasi"]):
    a.set_title(t); a.set_xlabel("epoch"); a.set_ylim(0, 1.02); a.grid(alpha=.3); a.legend()
fig.suptitle("MobileNetV3-Small: 3 mode pelatihan"); fig.tight_layout()
fig.savefig("results/akurasi_per_epoch.png", dpi=150)
