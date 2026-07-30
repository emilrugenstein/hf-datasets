"""Build the standalone AI World distribution viz (dataset popularity by org type).

Latest weekly snapshot only, reusing the loaders from `hf_org_trends_build_viewer`
(same org-type join rules). Three switchable views over the same two metrics
(all-time downloads, likes), one panel per metric:

  box    — box plots per org type. The distributions are zero-inflated (~92% of
           datasets have 0 likes), so quantiles cover datasets with >=1 of the
           metric and the zero mass is carried as a separate "share >= 1" number;
           the log axis then starts at 1.
  share  — quantile-share stacked bars: how much of the type's total volume its
           top 1% / next 9% / next 40% / bottom 50% of datasets carry (computed
           over ALL datasets of the type; zeros add nothing to volume).
  ridge  — ridgeline of the log10 density of datasets with >=1, per type, each
           curve normalised to equal peak height. Grid capped at the pooled
           99.9th percentile so the extreme tail doesn't stretch the axis.
  ridgew — ridgeline of volume by ACCOUNT rank percentile: authors get their
           datasets' metric aggregated and are ranked within their type (linear
           x, best accounts right); curve = share of the type's total volume
           contributed there, on a log height scale. One shared scale per panel
           (higher = more of the type's volume). A pill zooms into the top 10%.
  size   — single-panel ridgeline of the log10 density of repo size in bytes
           (`mainSize`, ~99.9% coverage), per type, peak-normalised, grid capped
           at the pooled 0.1th–99.9th percentiles.

Org types are ordered once, by median all-time downloads, so all views compare.

The JSON payload replaces /*__DATA__*/ in `viewer/datasets_box_viz_template.html`
and is written to `viewer/hf_datasets_box_viz.html` — one self-contained file in
the AI World house style, ready to host at viz.aiworld.eu.

Usage:
  python hf_datasets_box_build_viz.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

import hf_org_trends_build_viewer as base

TEMPLATE = Path("viewer/datasets_box_viz_template.html")
OUT_HTML = Path("viewer/hf_datasets_box_viz.html")

METRICS = {"dlat": "downloadsAllTime", "lk": "likes"}
QUANTS = [("p5", 0.05), ("q1", 0.25), ("med", 0.5), ("q3", 0.75), ("p95", 0.95), ("p99", 0.99)]
DENS_BINS = 80
DENS_SIGMA = 0.12  # KDE bandwidth in log10 units; wide enough to blur the likes integer comb
RANK_SIGMA = 0.008  # rank-percentile bandwidth (0.8% of the catalog) for the volume-by-rank view

log = logging.getLogger("hf_datasets_box_build_viz")


def box_stats(df: pl.DataFrame, col: str) -> dict[str, dict]:
    """Per org type: dataset count, share with >=1, and box quantiles over that subset."""
    counts = df.group_by("org_type").agg(pl.len().alias("n"), (pl.col(col) >= 1).mean().alias("share"))
    aggs = [pl.col(col).quantile(q).alias(k) for k, q in QUANTS]
    aggs += [pl.col(col).mean().alias("mean"), pl.col(col).max().alias("max")]
    sub = df.filter(pl.col(col) >= 1).group_by("org_type").agg(aggs)
    return {r["org_type"]: r for r in counts.join(sub, on="org_type").iter_rows(named=True)}


def concentration(vals: np.ndarray) -> dict[str, float]:
    """Cumulative volume share of the top 1% / 10% / 50% of datasets (all datasets, zeros incl.)."""
    v = np.sort(vals)[::-1]
    cum = np.cumsum(v)
    total = cum[-1]
    share = lambda p: round(float(cum[max(1, math.ceil(p * len(v))) - 1] / total), 4)
    return {"c1": share(0.01), "c10": share(0.10), "c50": share(0.50)}


def _kernel(sigma_bins: float) -> np.ndarray:
    r = int(math.ceil(4 * sigma_bins))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_bins) ** 2)
    return k / k.sum()


def densities(df: pl.DataFrame, col: str, types: list[str]) -> dict:
    """Per org type: smoothed histogram of log10(value) over datasets with >=1,
    on a shared grid capped at the pooled 99.9th percentile, peak-normalised."""
    logv = df.filter(pl.col(col) >= 1).select("org_type", pl.col(col).log(10).alias("x"))
    hi = float(logv["x"].quantile(0.999))
    edges = np.linspace(0.0, hi, DENS_BINS + 1)
    kernel = _kernel(DENS_SIGMA / (hi / DENS_BINS))  # log-units -> bins
    out = []
    for t in types:
        x = logv.filter(pl.col("org_type") == t)["x"].to_numpy()
        h, _ = np.histogram(x[x <= hi], bins=edges)
        d = np.convolve(h.astype(float), kernel, mode="same")
        out.append([round(float(v), 3) for v in d / d.max()])
    return {"hi": round(hi, 3), "d": out}


def size_densities(df: pl.DataFrame, types: list[str]) -> dict:
    """Per org type: smoothed histogram of log10(repo size in bytes) over datasets
    with a known positive `mainSize` (~99.9% coverage), on a pooled 0.1th–99.9th
    percentile grid, peak-normalised. Ships per-type coverage and byte quantiles
    for the axis labels / tooltip."""
    sized = df.filter(pl.col("mainSize") > 0)
    logv = sized.select("org_type", pl.col("mainSize").log(10).alias("x"))
    lo, hi = float(logv["x"].quantile(0.001)), float(logv["x"].quantile(0.999))
    edges = np.linspace(lo, hi, DENS_BINS + 1)
    kernel = _kernel(DENS_SIGMA / ((hi - lo) / DENS_BINS))
    dens = []
    for t in types:
        x = logv.filter(pl.col("org_type") == t)["x"].to_numpy()
        h, _ = np.histogram(x[(x >= lo) & (x <= hi)], bins=edges)
        d = np.convolve(h.astype(float), kernel, mode="same")
        dens.append([round(float(v), 3) for v in d / d.max()])
    cover = {r["org_type"]: r["c"] for r in
             df.group_by("org_type").agg((pl.col("mainSize") > 0).mean().alias("c")).iter_rows(named=True)}
    aggs = [pl.col("mainSize").quantile(q).alias(k) for k, q in QUANTS]
    aggs += [pl.col("mainSize").mean().alias("mean"), pl.col("mainSize").max().alias("max")]
    qs = {r["org_type"]: r for r in sized.group_by("org_type").agg(aggs).iter_rows(named=True)}
    return {
        "lo": round(lo, 3), "hi": round(hi, 3), "d": dens,
        "share": [round(cover[t], 4) for t in types],
        **{k: [int(qs[t][k]) for t in types] for k in [k for k, _ in QUANTS] + ["mean", "max"]},
    }


def rank_densities(by_author: pl.DataFrame, types: list[str], width: float = 1.0) -> dict:
    """Per org type: smoothed share of the type's total volume by ACCOUNT rank
    percentile — accounts are the type's authors with their datasets' metric
    aggregated, x = top fraction (LINEAR 0..width, best accounts first), over
    ALL of the type's accounts (zero-volume accounts rank bottom with no
    weight). Heights are shares of the type's TOTAL volume, normalised by the
    global max so one scale keeps types comparable; per-bin shares span orders
    of magnitude, so the template draws them on a LOG height axis (hence 6
    decimals). width < 1 zooms the grid into the top slice at full bin
    resolution; the kernel scales with the domain, keeping the relative
    smoothing identical."""
    edges = np.linspace(0.0, width, DENS_BINS + 1)
    kernel = _kernel(RANK_SIGMA * DENS_BINS)  # sigma is a fraction of the domain width
    raw = []
    for t in types:
        v = np.sort(by_author.filter(pl.col("org_type") == t)["v"].to_numpy())[::-1]
        nz = v[v >= 1].astype(float)
        x = (np.arange(len(nz)) + 0.5) / len(v)  # top-fraction of each ranked account
        keep = x <= width
        h, _ = np.histogram(x[keep], bins=edges, weights=nz[keep] / nz.sum())
        raw.append(np.convolve(h, kernel, mode="same"))
    gmax = max(d.max() for d in raw)
    return {"w": width, "d": [[round(float(v) / gmax, 6) for v in d] for d in raw]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_HTML, help="Output HTML path.")
    ap.add_argument("--orgs-csv", type=Path, default=None, help="Org scrape CSV (default: newest match).")
    args = ap.parse_args()

    if args.orgs_csv is None:
        args.orgs_csv = base.newest_orgs_csv()
    if args.orgs_csv is None or not args.orgs_csv.exists():
        log.error("No orgs CSV found (looked for %s)", base.ORGS_GLOB)
        return 1

    snapshots = base.list_snapshots("datasets")
    if not snapshots:
        log.error("No snapshot files in %s", base.DATA_DIRS["datasets"])
        return 1
    date_str, path = snapshots[-1]
    log.info("Latest snapshot: %s", date_str)

    orgs = base.load_orgs(args.orgs_csv)
    df = base.read_week(path, orgs)
    if df["downloadsAllTime"].null_count() == len(df):
        log.error("Snapshot %s has no all-time downloads — aborting", date_str)
        return 1
    log.info("Snapshot rows: %d", len(df))

    stats = {m: box_stats(df, col) for m, col in METRICS.items()}
    # one shared type order (median all-time downloads, desc) keeps all views comparable
    types = sorted(base.ORG_TYPES, key=lambda t: stats["dlat"][t]["med"], reverse=True)

    n_orgs = {r["org_type"]: r["n"] for r in
              df.group_by("org_type").agg(pl.col("author").n_unique().alias("n")).iter_rows(named=True)}

    metrics: dict[str, dict] = {}
    for m, col in METRICS.items():
        conc = {t: concentration(df.filter(pl.col("org_type") == t)[col].fill_null(0).to_numpy()) for t in types}
        # the rank views work on accounts: each author's datasets aggregated
        by_author = df.group_by("org_type", "author").agg(pl.col(col).fill_null(0).sum().alias("v"))
        conco = {t: concentration(by_author.filter(pl.col("org_type") == t)["v"].to_numpy()) for t in types}
        metrics[m] = {
            "share": [round(stats[m][t]["share"], 4) for t in types],
            **{k: [int(stats[m][t][k]) for t in types] for k, _ in QUANTS},
            "max": [int(stats[m][t]["max"]) for t in types],
            "mean": [round(stats[m][t]["mean"], 1) for t in types],
            "total": [int(df.filter(pl.col("org_type") == t)[col].sum()) for t in types],
            "conc": {k: [conc[t][k] for t in types] for k in ("c1", "c10", "c50")},
            "conco": {k: [conco[t][k] for t in types] for k in ("c1", "c10", "c50")},
            "dens": densities(df, col, types),
            "densr": rank_densities(by_author, types),
            "densrz": rank_densities(by_author, types, width=0.1),
        }
        log.info("[%s] top-1%% volume share — datasets: %s · accounts: %s", m,
                 {t: conc[t]["c1"] for t in types}, {t: conco[t]["c1"] for t in types})

    payload = {
        "built": date.today().isoformat(),
        "snapshot": date_str,
        "org_types": types,
        "n": [stats["dlat"][t]["n"] for t in types],
        "n_orgs": [n_orgs[t] for t in types],
        "metrics": metrics,
        "size": size_densities(df, types),
    }

    template = TEMPLATE.read_text(encoding="utf-8")
    html = template.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")), 1)
    tmp = args.out.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, args.out)
    log.info("Wrote %s (%.1f KB)", args.out, len(html) / 1e3)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    sys.exit(main())
