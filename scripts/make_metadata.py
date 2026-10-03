"""Buat dataset_raw/metadata.csv. Kolom tanggal & kondisi_cahaya diisi manual."""
import csv
from pathlib import Path
rows = [(f.name, c.name, "", "") for c in sorted(Path("dataset_raw").iterdir()) if c.is_dir()
        for f in sorted(c.glob("*.*"))]
with open("dataset_raw/metadata.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["nama_file", "kelas", "tanggal", "kondisi_cahaya"]); w.writerows(rows)
print(len(rows), "baris")
