#!/usr/bin/env bash
set -e
python scripts/split.py
for m in feature partial scratch; do python scripts/train.py --mode $m --epochs 10; done
python scripts/latency.py
python scripts/report.py
