"""Run the six stages back to back with one set of settings.

    python scripts/run_all.py --profile quick
    python scripts/run_all.py --config config.json
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STAGES = [
    "01_preprocess.py",
    "02_subtype_networks.py",
    "03_rewiring.py",
    "04_stability_modules.py",
    "05_survival.py",
    "06_metabric_validation.py",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--profile", choices=["full", "quick"], default="full")
    ap.add_argument("--from-stage", type=int, default=1)
    ap.add_argument("--to-stage", type=int, default=len(STAGES))
    args = ap.parse_args()

    common = ["--profile", args.profile] + (["--config", args.config] if args.config else [])
    for i, stage in enumerate(STAGES, start=1):
        if not args.from_stage <= i <= args.to_stage:
            continue
        print(f"\n===== {stage} =====", flush=True)
        started = time.time()
        code = subprocess.call([sys.executable, os.path.join(HERE, stage)] + common)
        if code != 0:
            raise SystemExit(f"{stage} failed with exit code {code}")
        print(f"===== {stage} done in {time.time() - started:.0f}s =====", flush=True)


if __name__ == "__main__":
    main()
