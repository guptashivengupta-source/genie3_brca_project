"""Reading and writing the tables the pipeline passes between stages."""

from __future__ import annotations

import gzip
import os
from typing import Iterable, List

import pandas as pd

from .config import ROOT


def data_path(*parts: str) -> str:
    """GENIE3_DATA lets a test or a second cohort point somewhere other than data/."""
    return os.path.join(os.environ.get("GENIE3_DATA", os.path.join(ROOT, "data")), *parts)


def results_path(*parts: str) -> str:
    root = os.environ.get("GENIE3_RESULTS", os.path.join(ROOT, "results"))
    path = os.path.join(root, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def _opener(path: str):
    return gzip.open if path.endswith(".gz") else open


def read_matrix(path: str) -> pd.DataFrame:
    """Genes in rows, samples in columns, first column is the gene symbol."""
    with _opener(path)(path, "rt", encoding="utf-8") as fh:
        df = pd.read_csv(fh, sep="\t", index_col=0)
    df.index = df.index.astype(str).str.strip()
    df.columns = df.columns.astype(str).str.strip()
    return df.loc[~df.index.duplicated(keep="first")].astype(float)


def read_table(path: str, index_col=None) -> pd.DataFrame:
    with _opener(path)(path, "rt", encoding="utf-8") as fh:
        return pd.read_csv(fh, sep="\t", index_col=index_col, low_memory=False)


def write_table(df: pd.DataFrame, path: str, index: bool = True) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=index)
    return path


def read_gene_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]


def write_gene_list(genes: Iterable[str], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(genes) + "\n")
    return path

