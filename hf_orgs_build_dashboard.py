"""Build the interactive organisation dashboard from weekly snapshot minima.

Aggregates each weekly snapshot by account (`author`) for both hubs:
  - datasets: data/hf_datasets_historic/hf_datasets_min_<date>.parquet (this repo)
  - models:   hf_main_min_<date>.parquet from the old capstone project
    (--models-dir; used as-is for this build, ends 2026-04-22)
plus a combined mode covering the weeks both hubs share.

Per mode it keeps every account that was ever in a weekly top-POOL by 30-day
downloads or by weekly like gain (likes delta), embeds their weekly series
(downloads + cumulative likes; deltas are derived client-side so they stay
exact), and hub-wide totals for the concentration section. POOL (default 500)
is the ceiling for the UI's top-N selector, so raising the UI beyond top 100
later needs no rebuild as long as N <= POOL.

Account org_type/followers come from the newest data/hf_orgs scrape, falling
back to the org_type baked into the weekly files.

The JSON payload replaces /*__DATA__*/ in viewer/org_dashboard_template.html
and is written to viewer/hf_orgs_dashboard.html (self-contained, no server).

Usage:
  python hf_orgs_build_dashboard.py             # full build
  python hf_orgs_build_dashboard.py --quick 8   # every 8th week, fast iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from datetime import date
from pathlib import Path

import polars as pl
from dotenv import load_dotenv

SNAPSHOT_REPO = "hfmlsoc/hub_weekly_snapshots"
DS_DIR = Path("data/hf_datasets_historic")
MODELS_DIR = Path(r"C:\Users\mail\OneDrive\Dokumente\StudiumLUC\Capstone\Programming\data\hf_main_historic")
ORGS_DIR = Path("data/hf_orgs")
TEMPLATE = Path("viewer/org_dashboard_template.html")
OUT_HTML = Path("viewer/hf_orgs_dashboard.html")

TOP_POOL = 500  # per-week per-metric selection pool = max top-N the UI can show

DS_RE = re.compile(r"hf_datasets_min_(\d{4}-\d{2}-\d{2})\.parquet$")
MD_RE = re.compile(r"hf_main_min_(\d{4}-\d{2}-\d{2})\.parquet$")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf_orgs_build_dashboard")


def list_snapshots(directory: Path, pattern: re.Pattern) -> list[tuple[str, Path]]:
    pairs = [(m.group(1), p) for p in directory.glob("*.parquet") if (m := pattern.search(p.name))]
    return sorted(pairs)


def load_week_agg(path: Path) -> pl.DataFrame:
    """One week of one hub, aggregated per account: downloads, likes, repos, org_type."""
    lf = pl.scan_parquet(path)
    cols = lf.collect_schema().names()
    org_expr = (
        pl.col("org_type").drop_nulls().first() if "org_type" in cols else pl.lit(None, dtype=pl.String)
    ).alias("org_type")
    dlat_expr = (
        pl.col("downloadsAllTime").sum() if "downloadsAllTime" in cols else pl.lit(0, dtype=pl.Int64)
    ).alias("dlat")
    return (
        lf.drop_nulls("author")
        .group_by("author")
        .agg(
            pl.col("downloads").sum().alias("dl"),
            pl.col("likes").sum().alias("lk"),
            dlat_expr,
            pl.len().alias("n"),
            org_expr,
        )
        .collect()
    )


def combine_aggs(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    """Element-wise sum of two per-account aggregates (full outer, missing = 0)."""
    joined = a.join(b, on="author", how="full", coalesce=True, suffix="_b")
    return joined.select(
        "author",
        (pl.col("dl").fill_null(0) + pl.col("dl_b").fill_null(0)).alias("dl"),
        (pl.col("lk").fill_null(0) + pl.col("lk_b").fill_null(0)).alias("lk"),
        (pl.col("dlat").fill_null(0) + pl.col("dlat_b").fill_null(0)).alias("dlat"),
        (pl.col("n").fill_null(0) + pl.col("n_b").fill_null(0)).alias("n"),
    )


def hub_totals(agg: pl.DataFrame) -> dict:
    return {
        "dl": int(agg["dl"].sum() or 0),
        "lk": int(agg["lk"].sum() or 0),
        "dlat": int(agg["dlat"].sum() or 0),  # 0 before the 2025-02-26 tracking boundary
        "n_repo": int(agg["n"].sum() or 0),
        "n_auth": len(agg),
    }


def select_top(agg: pl.DataFrame, prev: pl.DataFrame | None, pool: int) -> set[str]:
    """Weekly top-pool accounts by 30-day/all-time downloads and by like gain vs previous week."""
    sel = set(agg.top_k(pool, by="dl")["author"])
    if agg["dlat"].sum():
        sel |= set(agg.top_k(pool, by="dlat")["author"])
    if prev is not None:
        delta = (
            agg.select("author", "lk")
            .join(prev.select("author", lk_prev="lk"), on="author", how="inner")
            .with_columns((pl.col("lk") - pl.col("lk_prev")).alias("lkd"))
        )
        sel |= set(delta.top_k(pool, by="lkd")["author"])
    return sel


def round_sig(v: int | None, sig: int = 3) -> int | None:
    """Round to `sig` significant figures (download series only, for compactness)."""
    if v is None or v == 0:
        return v
    return int(round(v, sig - int(math.floor(math.log10(abs(v)))) - 1))


def trim_series(values: list) -> tuple[int, list]:
    """Trim leading/trailing nulls; returns (start_index, trimmed) — (0, []) if all null."""
    first = next((i for i, v in enumerate(values) if v is not None), None)
    if first is None:
        return 0, []
    last = max(i for i, v in enumerate(values) if v is not None)
    return first, values[first : last + 1]


def collect_series(aggs: list[pl.DataFrame], selected: set[str]) -> dict[str, dict]:
    """Weekly (downloads, all-time downloads, cumulative likes, repo count) arrays per selected account."""
    n_weeks = len(aggs)
    records: dict[str, dict] = {
        a: {"dl": [None] * n_weeks, "lk": [None] * n_weeks, "n": [None] * n_weeks, "dlat": [None] * n_weeks}
        for a in selected
    }
    sel_series = pl.Series("author", sorted(selected)).implode()
    for week, agg in enumerate(aggs):
        has_dlat = bool(agg["dlat"].sum())  # all-time downloads only tracked from 2025-02-26
        rows = agg.filter(pl.col("author").is_in(sel_series)).select("author", "dl", "lk", "n", "dlat")
        for author, dl, lk, n, dlat in rows.iter_rows():
            rec = records[author]
            rec["dl"][week] = round_sig(int(dl))
            rec["lk"][week] = int(lk)
            rec["n"][week] = int(n)
            rec["dlat"][week] = round_sig(int(dlat)) if has_dlat else None
    return records


def load_org_meta() -> dict[str, tuple[str, int | None]]:
    """slug -> (org_type, followers) from the newest orgs scrape."""
    scrapes = sorted(ORGS_DIR.glob("hf_orgs_scraped_*.csv"))
    if not scrapes:
        log.warning("No orgs scrape found in %s", ORGS_DIR)
        return {}
    latest = scrapes[-1]
    df = pl.read_csv(latest, columns=["slug", "org_type", "followers"])
    log.info("Org metadata from %s (%d orgs)", latest.name, len(df))
    return {r["slug"]: (r["org_type"], r["followers"]) for r in df.iter_rows(named=True)}


def load_org_countries() -> dict[str, str]:
    """slug -> ISO country code from the newest classification file (LLM-assisted,
    covers the top 500 by dataset downloads at its build date; everything else is unknown)."""
    files = sorted(ORGS_DIR.glob("org_countries_*.csv"))
    if not files:
        log.warning("No org_countries file in %s — all accounts will be 'unknown'", ORGS_DIR)
        return {}
    latest = files[-1]
    df = pl.read_csv(latest, columns=["account", "country"])
    out = {r["account"]: r["country"] for r in df.iter_rows(named=True) if r["country"] != "unknown"}
    log.info("Org countries from %s (%d classified)", latest.name, len(out))
    return out


def load_size_stats(date_str: str, pool: set[str]) -> dict | None:
    """Download-volume proxy for the datasets hub at one week, read remotely.

    `mainSize` (bytes) only lives in the raw snapshots (not the min files), so this
    reads the needed columns from the hub. volume = 30-day downloads x mainSize —
    an upper bound that assumes every download event pulls the full dataset.
    """
    uri = f"hf://datasets/{SNAPSHOT_REPO}/datasets/{date_str}/datasets.parquet"
    try:
        df = pl.scan_parquet(uri).select("author", "downloads", "mainSize").collect()
    except Exception as exc:  # offline / schema era without mainSize — dashboard just omits the tile
        log.warning("Size stats unavailable (%s) — building without the volume tile", exc)
        return None
    covered = df.drop_nulls()
    dl_cov = (covered["downloads"].sum() or 0) / max(1, df["downloads"].sum() or 0)
    per = (
        covered.group_by("author")
        .agg((pl.col("downloads") * pl.col("mainSize")).sum().alias("bytes"))
        .filter(pl.col("author").is_in(pl.Series(sorted(pool)).implode()) & (pl.col("bytes") > 0))
    )
    hub_bytes = float((covered["downloads"] * covered["mainSize"]).sum())
    log.info("Size stats %s: hub volume %.0f PB, %.1f%% of downloads covered, %d pool accounts",
             date_str, hub_bytes / 1e15, dl_cov * 100, len(per))
    return {
        "week": date_str,
        "hub_bytes": float(f"{hub_bytes:.4g}"),
        "dl_cov": round(dl_cov, 4),
        "acct": {a: float(f"{b:.3g}") for a, b in per.iter_rows()},
    }


def process_hub(name: str, snapshots: list[tuple[str, Path]], pool: int):
    """Pass 1 for one hub: per-week aggregates (cached), hub totals, selection, org types."""
    aggs, hub, selected, org_types = [], [], set(), {}
    prev = None
    for date_str, path in snapshots:
        agg = load_week_agg(path)
        for author, org_type in agg.drop_nulls("org_type").select("author", "org_type").iter_rows():
            org_types[author] = org_type  # latest week wins
        agg = agg.drop("org_type")
        hub.append(hub_totals(agg))
        selected |= select_top(agg, prev, pool)
        aggs.append(agg)
        prev = agg
        log.info("[%s %s] accounts=%d selected so far=%d", name, date_str, len(agg), len(selected))
    return aggs, hub, selected, org_types


def build_payload(master_dates, modes, org_meta, org_types_baked, pool, size_stats, org_countries) -> dict:
    date_idx = {d: i for i, d in enumerate(master_dates)}

    all_orgs = sorted(set().union(*(m["records"].keys() for m in modes.values())))
    types = []
    type_idx: dict[str, int] = {}
    countries = ["unknown"]
    country_idx: dict[str, int] = {"unknown": 0}
    orgs_out = []
    for author in all_orgs:
        org_type, followers = org_meta.get(author, (None, None))
        org_type = org_type or org_types_baked.get(author) or "individual"
        if org_type not in type_idx:
            type_idx[org_type] = len(types)
            types.append(org_type)
        country = org_countries.get(author, "unknown")
        if country not in country_idx:
            country_idx[country] = len(countries)
            countries.append(country)
        entry = [author, type_idx[org_type], followers]
        for key in ("ds", "md", "cb"):
            rec = modes[key]["records"].get(author)
            if rec is None:
                entry.append(0)
                continue
            dl_start, dl_vals = trim_series(rec["dl"])
            lk_start, lk_vals = trim_series(rec["lk"])
            n_vals = rec["n"][dl_start : dl_start + len(dl_vals)]  # present exactly when dl is
            dlat_vals = rec["dlat"][dl_start : dl_start + len(dl_vals)]  # dl-aligned, null pre-boundary
            entry.append([dl_start, dl_vals, lk_start, lk_vals, n_vals, dlat_vals])
        entry.append(country_idx[country])
        orgs_out.append(entry)

    return {
        "built": date.today().isoformat(),
        "pool": pool,
        "dates": master_dates,
        "modes": {
            key: {
                "label": m["label"],
                "weeks": [date_idx[d] for d in m["dates"]],
                "hub": {k: [h[k] for h in m["hub"]] for k in ("dl", "lk", "n_repo", "n_auth")}
                | {"dlat": [h["dlat"] or None for h in m["hub"]]},
                "at0": next((i for i, h in enumerate(m["hub"]) if h["dlat"]), None),
            }
            for key, m in modes.items()
        },
        "org_types": types,
        "countries": countries,
        "orgs": orgs_out,
        "size": size_stats,
    }


def render_html(payload: dict, out_path: Path) -> None:
    template = TEMPLATE.read_text(encoding="utf-8")
    data_json = json.dumps(payload, separators=(",", ":"))
    html = template.replace("/*__DATA__*/", data_json, 1)
    tmp = out_path.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, out_path)
    log.info("Wrote %s (%.1f MB, data %.1f MB)", out_path, len(html) / 1e6, len(data_json) / 1e6)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Directory with hf_main_min_<date>.parquet files.")
    ap.add_argument("--out", type=Path, default=OUT_HTML, help="Output HTML path.")
    ap.add_argument("--top-pool", type=int, default=TOP_POOL, help="Per-week per-metric selection pool (max UI top-N).")
    ap.add_argument("--quick", type=int, default=None, metavar="N", help="Use every Nth week (always incl. latest).")
    ap.add_argument("--skip-size", action="store_true", help="Skip the remote mainSize read (no volume tile).")
    args = ap.parse_args()
    load_dotenv()

    ds_snaps = list_snapshots(DS_DIR, DS_RE)
    md_snaps = list_snapshots(args.models_dir, MD_RE)
    if not ds_snaps or not md_snaps:
        log.error("Missing snapshots: %d dataset weeks, %d model weeks", len(ds_snaps), len(md_snaps))
        return 1
    if args.quick:
        def thin(snaps):
            kept = snaps[:: args.quick]
            return kept + ([snaps[-1]] if snaps[-1] not in kept else [])
        ds_snaps, md_snaps = thin(ds_snaps), thin(md_snaps)
    log.info("Datasets: %d weeks (%s … %s)", len(ds_snaps), ds_snaps[0][0], ds_snaps[-1][0])
    log.info("Models:   %d weeks (%s … %s)", len(md_snaps), md_snaps[0][0], md_snaps[-1][0])

    ds_aggs, ds_hub, ds_sel, ds_types = process_hub("ds", ds_snaps, args.top_pool)
    md_aggs, md_hub, md_sel, md_types = process_hub("md", md_snaps, args.top_pool)

    # combined mode: only weeks both hubs have, element-wise account sums
    ds_by_date = {d: i for i, (d, _) in enumerate(ds_snaps)}
    md_by_date = {d: i for i, (d, _) in enumerate(md_snaps)}
    cb_dates = sorted(set(ds_by_date) & set(md_by_date))
    cb_aggs, cb_hub, cb_sel = [], [], set()
    prev = None
    for d in cb_dates:
        agg = combine_aggs(ds_aggs[ds_by_date[d]], md_aggs[md_by_date[d]])
        cb_hub.append(hub_totals(agg))
        cb_sel |= select_top(agg, prev, args.top_pool)
        cb_aggs.append(agg)
        prev = agg
        log.info("[cb %s] accounts=%d selected so far=%d", d, len(agg), len(cb_sel))

    modes = {
        "ds": {"label": "Datasets", "dates": [d for d, _ in ds_snaps], "hub": ds_hub,
               "records": collect_series(ds_aggs, ds_sel)},
        "md": {"label": "Models", "dates": [d for d, _ in md_snaps], "hub": md_hub,
               "records": collect_series(md_aggs, md_sel)},
        "cb": {"label": "Combined", "dates": cb_dates, "hub": cb_hub,
               "records": collect_series(cb_aggs, cb_sel)},
    }
    master_dates = sorted(set(ds_by_date) | set(md_by_date))

    size_stats = None if args.skip_size else load_size_stats(ds_snaps[-1][0], ds_sel)

    org_types_baked = {**md_types, **ds_types}  # datasets scrape merge is newer, wins
    payload = build_payload(master_dates, modes, load_org_meta(), org_types_baked, args.top_pool, size_stats,
                            load_org_countries())
    log.info(
        "Payload: %d accounts (ds %d / md %d / cb %d), %d master weeks",
        len(payload["orgs"]), len(ds_sel), len(md_sel), len(cb_sel), len(master_dates),
    )

    render_html(payload, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
