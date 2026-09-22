"""Fetch TCGA-BRCA from UCSC Xena and METABRIC from the cBioPortal API.

    python scripts/download_data.py tcga
    python scripts/download_data.py metabric --genes results/network_genes.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.io_utils import data_path, read_gene_list, write_table

XENA = "https://tcga.xenahubs.net/download"
TCGA_FILES = {
    "HiSeqV2.gz": f"{XENA}/TCGA.BRCA.sampleMap/HiSeqV2.gz",
    "BRCA_clinicalMatrix.tsv": f"{XENA}/TCGA.BRCA.sampleMap/BRCA_clinicalMatrix",
    "BRCA_survival.tsv": f"{XENA}/survival/BRCA_survival.txt",
}

CBIO = "https://www.cbioportal.org/api"
METABRIC_STUDY = "brca_metabric"
METABRIC_PROFILE = "brca_metabric_mrna"
METABRIC_SAMPLE_LIST = "brca_metabric_all"
DAYS_PER_MONTH = 30.4375


def _get(url: str, timeout: int = 120):
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.load(fh)


def _post(url: str, payload, timeout: int = 300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


def download_tcga(force: bool = False) -> None:
    for name, url in TCGA_FILES.items():
        dest = data_path("tcga", name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and not force:
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}")
        urllib.request.urlretrieve(url, dest)
        print(f"       {os.path.getsize(dest) / 1e6:.1f} MB")


def _metabric_clinical() -> pd.DataFrame:
    frames = []
    for kind in ("PATIENT", "SAMPLE"):
        url = (f"{CBIO}/studies/{METABRIC_STUDY}/clinical-data"
               f"?clinicalDataType={kind}&projection=SUMMARY&pageSize=10000000")
        records = _get(url)
        key = "patientId" if kind == "PATIENT" else "sampleId"
        df = pd.DataFrame(records)
        frames.append(df.pivot_table(index=key, columns="clinicalAttributeId",
                                     values="value", aggfunc="first"))
    clinical = frames[0].join(frames[1], how="outer", lsuffix="", rsuffix="_sample")
    clinical.index.name = "sample"

    months = pd.to_numeric(clinical.get("OS_MONTHS"), errors="coerce")
    status = clinical.get("OS_STATUS", pd.Series(dtype=object)).astype(str)
    clinical["OS"] = status.str.startswith("1").astype(float).where(status.ne("nan"))
    clinical["OS.time"] = months * DAYS_PER_MONTH

    rfs_months = pd.to_numeric(clinical.get("RFS_MONTHS"), errors="coerce")
    rfs_status = clinical.get("RFS_STATUS", pd.Series(dtype=object)).astype(str)
    clinical["DFI"] = rfs_status.str.startswith("1").astype(float).where(rfs_status.ne("nan"))
    clinical["DFI.time"] = rfs_months * DAYS_PER_MONTH
    return clinical


def _entrez_ids(symbols):
    mapped = _post(f"{CBIO}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL", list(symbols))
    return {g["entrezGeneId"]: g["hugoGeneSymbol"] for g in mapped}


def download_metabric(genes_file: str, chunk: int = 400) -> None:
    symbols = read_gene_list(genes_file)
    print(f"[get ] METABRIC expression for {len(symbols)} genes")
    lookup = _entrez_ids(symbols)
    missing = sorted(set(symbols) - set(lookup.values()))
    if missing:
        print(f"       {len(missing)} symbols not in cBioPortal, e.g. {missing[:5]}")

    ids = sorted(lookup)
    frames = []
    for start in range(0, len(ids), chunk):
        batch = ids[start:start + chunk]
        rows = _post(
            f"{CBIO}/molecular-profiles/{METABRIC_PROFILE}/molecular-data/fetch"
            "?projection=SUMMARY",
            {"entrezGeneIds": batch, "sampleListId": METABRIC_SAMPLE_LIST},
        )
        frames.append(pd.DataFrame(rows)[["entrezGeneId", "sampleId", "value"]])
        print(f"       {min(start + chunk, len(ids))}/{len(ids)} genes")

    long = pd.concat(frames, ignore_index=True)
    long["gene"] = long["entrezGeneId"].map(lookup)
    expr = long.pivot_table(index="gene", columns="sampleId", values="value",
                            aggfunc="mean").dropna(how="all")
    write_table(expr, data_path("metabric", "expression.tsv"))
    print(f"       expression {expr.shape[0]} genes x {expr.shape[1]} samples")

    clinical = _metabric_clinical()
    clinical = clinical.loc[clinical.index.intersection(expr.columns)]
    write_table(clinical, data_path("metabric", "clinical.tsv"))
    print(f"       clinical {clinical.shape[0]} samples x {clinical.shape[1]} fields")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=["tcga", "metabric"])
    ap.add_argument("--genes", help="gene list for the METABRIC fetch")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    try:
        if args.source == "tcga":
            download_tcga(args.force)
        else:
            if not args.genes:
                ap.error("metabric needs --genes (run scripts/01_preprocess.py first)")
            download_metabric(args.genes)
    except urllib.error.URLError as exc:
        raise SystemExit(f"download failed ({exc}); see data/README.md for manual links")


if __name__ == "__main__":
    main()
