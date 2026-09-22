"""Pipeline parameters, overridable from config.json."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # gene selection
    min_count: float = 1.0
    min_samples: int = 20
    n_network_genes: int = 500

    # subtypes kept for network inference
    subtypes: List[str] = field(default_factory=lambda: ["LumA", "LumB", "Her2", "Basal"])
    min_subtype_n: int = 100

    # composition adjustment
    adjust_composition: bool = True
    composition_axes: List[str] = field(
        default_factory=lambda: ["adipocyte", "stromal", "immune"]
    )

    # GENIE3
    tree_method: str = "RF"
    n_trees: int = 100
    max_features: str = "sqrt"
    n_jobs: int = -1
    seed: int = 1234
    tf_only: bool = False

    # sample-size matching
    n_match: int = 0          # 0 -> smallest retained subtype
    n_match_reps: int = 5

    # rewiring
    top_edges: int = 1000
    n_permutations: int = 10
    fdr: float = 0.05

    # bootstrap stability
    n_bootstrap: int = 30
    bootstrap_fraction: float = 0.8
    hub_top_k: int = 20
    hub_stability_threshold: float = 0.70

    # modules
    min_module_size: int = 12
    max_modules: int = 12
    module_stability_threshold: float = 0.50

    # survival
    survival_endpoint: str = "OS"
    max_followup_days: int = 3650
    covariates: List[str] = field(default_factory=lambda: ["age", "stage", "subtype"])

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        path = path or os.path.join(ROOT, "config.json")
        cfg = cls()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for key, val in json.load(fh).items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, val)
                    else:
                        raise KeyError(f"unknown config key: {key}")
        return cfg

    def quick(self) -> "Config":
        """Small-but-complete settings so the whole pipeline runs in minutes."""
        import copy

        c = copy.deepcopy(self)
        c.n_network_genes = 150
        c.n_trees = 50
        c.n_match_reps = 2
        c.n_permutations = 6
        c.n_bootstrap = 10
        c.top_edges = 300
        c.min_module_size = 6
        return c

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
