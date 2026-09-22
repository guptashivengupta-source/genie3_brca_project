"""Marker sets used to score tissue composition, plus the human TF list."""

from __future__ import annotations

import os
from typing import Dict, List

from .config import ROOT
from .io_utils import read_gene_list

# Provenance in data/ref/README.md.
COMPOSITION: Dict[str, List[str]] = {
    "adipocyte": [
        "ADIPOQ", "FABP4", "PLIN1", "PLIN4", "CIDEC", "CIDEA", "AQP7",
        "LEP", "LIPE", "CFD", "GPD1", "LPL", "PPARG", "TUSC5",
    ],
    "stromal": [
        "COL1A1", "COL1A2", "COL3A1", "COL5A1", "FN1", "FAP", "THY1",
        "POSTN", "SPARC", "DCN", "LUM", "ACTA2", "PDGFRB", "MMP2",
    ],
    "immune": [
        "PTPRC", "CD3D", "CD3E", "CD2", "CD8A", "CD19", "MS4A1", "CD68",
        "LYZ", "HLA-DRA", "IL2RG", "CXCL9", "GZMB", "ITGAM",
    ],
    "proliferation": [
        "MKI67", "CCNB1", "BUB1", "TOP2A", "AURKA", "RRM2", "TYMS",
        "CDK1", "UBE2C", "PLK1", "CCNE1", "MCM2",
    ],
}

_TF_CACHE: List[str] | None = None


def load_tfs(path: str | None = None) -> List[str]:
    """Human TFs from Lambert et al. 2018; empty list if the file is missing."""
    global _TF_CACHE
    if _TF_CACHE is None or path is not None:
        path = path or os.path.join(ROOT, "data", "ref", "tf_list.txt")
        tfs = read_gene_list(path) if os.path.exists(path) else []
        if path is None:
            return tfs
        _TF_CACHE = tfs
    return list(_TF_CACHE)


def is_tf(gene: str, tfs: List[str] | None = None) -> bool:
    return gene in set(tfs if tfs is not None else load_tfs())
