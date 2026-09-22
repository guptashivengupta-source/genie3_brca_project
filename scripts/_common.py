"""Shared argument handling for the numbered pipeline stages."""

from __future__ import annotations

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import Config  # noqa: E402

_START = time.time()


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", help="path to a config.json")
    ap.add_argument("--profile", choices=["full", "quick"], default="full",
                    help="quick shrinks every loop so the stage finishes in minutes")
    ap.add_argument("--seed", type=int)
    return ap


def get_config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    if args.profile == "quick":
        cfg = cfg.quick()
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    return cfg


def log(message: str) -> None:
    print(f"[{time.time() - _START:6.1f}s] {message}", flush=True)
