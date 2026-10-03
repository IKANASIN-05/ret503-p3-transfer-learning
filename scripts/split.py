"""Bagi dataset_raw -> data/train & data/val (80/20) berdasarkan URUTAN nama file.
Nama file berbasis waktu (mis. 0218091249.jpg) -> 20% terakhir (waktu paling akhir)
masuk val, sehingga frame beruntun tidak tersebar ke train dan val (cegah data leakage).
"""
import re, shutil
from pathlib import Path

RAW, OUT, VAL_FRAC = Path("dataset_raw"), Path("data"), 0.2
nat = lambda p: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]

if OUT.exists():
    shutil.rmtree(OUT)
for cls_dir in sorted(d for d in RAW.iterdir() if d.is_dir()):
    files = sorted(cls_dir.glob("*.*"), key=nat)
    n_val = max(1, round(len(files) * VAL_FRAC))
    for i, f in enumerate(files):
        split = "val" if i >= len(files) - n_val else "train"
        dst = OUT / split / cls_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst / f.name)
    print(f"{cls_dir.name}: {len(files) - n_val} train / {n_val} val")
