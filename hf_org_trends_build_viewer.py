"""Build the org-type trends HTML viewer from the weekly snapshot minima.

Two passes over `data/hf_datasets_historic/hf_datasets_min_<date>.parquet`:
  pass 1 — per-week per-org-type aggregates (downloads, all-time downloads,
           likes, counts, newly-added) plus top-N concentration sums and the
           per-week top author pools;
  pass 2 — weekly downloads/likes/count series for the union of "ever in a
           weekly top pool" authors (makes client-side top-100 ranking exact
           for any rank week and metric).

org_type is re-joined from the newest `data/hf_orgs/hf_orgs_scraped_*.csv`
(overriding the column baked into the parquets from an older scrape), with the
baked value as fallback for orgs missing from the CSV; unmatched authors count
as "individual". Caveat: one scrape date is applied to all historic snapshots.

The compact JSON payload replaces the /*__DATA__*/ placeholder in
`viewer/org_template.html` and is written to `viewer/hf_org_trends_viewer.html`
— a single self-contained file, no server needed. The payload has one slot per
entity ("datasets", "models"); entities without a local snapshot directory are
emitted as null and render as a disabled toggle. Adding models later only
requires filling data/hf_models_historic/ and passing --entities datasets,models.

Usage:
  python hf_org_trends_build_viewer.py             # full build
  python hf_org_trends_build_viewer.py --quick 8   # every 8th week, fast iteration
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

DATA_DIRS = {"datasets": Path("data/hf_datasets_historic")}
FILE_RES = {"datasets": re.compile(r"hf_datasets_min_(\d{4}-\d{2}-\d{2})\.parquet$")}
ENTITY_LABELS = {"datasets": "Datasets", "models": "Models"}
ORGS_GLOB = "data/hf_orgs/hf_orgs_scraped_*.csv"
TEMPLATE = Path("viewer/org_template.html")
OUT_HTML = Path("viewer/hf_org_trends_viewer.html")

ALLTIME_START = "2025-02-26"  # downloadsAllTime is null before this snapshot
TOP_N = 10                    # concentration overlay: top-N repos/orgs per org type
TOP_POOL = 500                # per-week per-metric author pool for the embedded orgs
                              # (>= the largest UI "top N accounts" option, so client-side ranking is exact)

# Canonical org-type order; colors live in the template keyed by these names.
ORG_TYPES = ["company", "university", "non-profit", "community", "classroom", "government", "individual"]

# log10 midpoint of each size_categories bucket (n = number of rows), e.g.
# "1K<n<10K" spans log10 3..4 -> 3.5. Datasets without the tag are excluded.
SIZE_LOG10_MID = {
    "n<1K": 1.5, "1K<n<10K": 3.5, "10K<n<100K": 4.5, "100K<n<1M": 5.5,
    "1M<n<10M": 6.5, "10M<n<100M": 7.5, "100M<n<1B": 8.5, "1B<n<10B": 9.5,
    "10B<n<100B": 10.5, "100B<n<1T": 11.5, "n>1T": 12.5,
}

# Upstream snapshots known to be identical to the prior week (row-count +
# likes-sum signature); a full build asserts detection matches this list.
KNOWN_FROZEN = {
    "datasets": ["2024-08-07", "2025-07-02", "2025-11-12", "2025-11-19", "2026-04-08", "2026-05-27", "2026-06-03"],
}

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf_org_trends_build_viewer")


def list_snapshots(entity: str) -> list[tuple[str, Path]]:
    """Sorted (date, path) pairs; the date list is the viewer's shared x-axis."""
    data_dir, file_re = DATA_DIRS[entity], FILE_RES[entity]
    pairs = [(m.group(1), p) for p in data_dir.glob("*.parquet") if (m := file_re.search(p.name))]
    return sorted(pairs)


def newest_orgs_csv() -> Path | None:
    files = sorted(Path().glob(ORGS_GLOB))  # date-stamped names sort chronologically
    return files[-1] if files else None


def load_orgs(csv_path: Path) -> pl.DataFrame:
    """author -> (csv_type, followers) from the org scrape, deduped on slug."""
    orgs = (
        pl.read_csv(csv_path, columns=["slug", "org_type", "followers"])
        .rename({"slug": "author", "org_type": "csv_type"})
        .unique(subset=["author"], keep="first")
    )
    log.info("Orgs CSV %s: %d orgs, %d with a type", csv_path.name, len(orgs), orgs["csv_type"].drop_nulls().len())
    return orgs


def read_week(path: Path, orgs: pl.DataFrame) -> pl.DataFrame:
    """One snapshot with the resolved org_type: CSV type -> baked type -> individual."""
    core = ["_id", "author", "likes", "downloads", "downloadsAllTime", "org_type", "tags"]
    available = pl.scan_parquet(path).collect_schema().names()
    df = pl.read_parquet(path, columns=core + (["mainSize"] if "mainSize" in available else []))
    if "mainSize" not in df.columns:  # only min files rebuilt after 2026-07 carry it
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("mainSize"))
    df = (
        df.rename({"org_type": "baked_type"})
        .join(orgs.select("author", "csv_type"), on="author", how="left")
        .with_columns(pl.coalesce("csv_type", "baked_type").fill_null("individual").alias("org_type"))
        .with_columns(
            pl.when(pl.col("org_type").is_in(ORG_TYPES))
            .then(pl.col("org_type"))
            .otherwise(pl.lit("individual"))
            .alias("org_type")
        )
        .drop("csv_type", "baked_type")
    )
    return df


def _per_type(frame: pl.DataFrame, value_col: str) -> list[int | None]:
    """Vector over ORG_TYPES from a (org_type, value) frame; missing types -> 0."""
    d = dict(frame.select("org_type", value_col).iter_rows())
    return [int(d.get(t, 0) or 0) for t in ORG_TYPES]


def _topn_per_type(df: pl.DataFrame, metric: str, n: int) -> list[int]:
    """Sum of the top-N repos per org type, on `metric`."""
    base = df.drop_nulls(metric).select("org_type", metric)
    top = base.filter(pl.col(metric).rank(method="ordinal", descending=True).over("org_type") <= n)
    return _per_type(top.group_by("org_type").agg(pl.col(metric).sum()), metric)


def _org_level_per_type(df: pl.DataFrame, metric: str | None, n: int) -> tuple[list[int], dict[str, list]]:
    """Per org type: top-N-organisation sum of `metric` (row count when None),
    plus the top-3 (author, value) rows."""
    agg = pl.len().alias("val") if metric is None else pl.col(metric).sum().alias("val")
    by = df.group_by("org_type", "author").agg(agg)
    ranked = by.with_columns(pl.col("val").rank(method="ordinal", descending=True).over("org_type").alias("rk"))
    topn_sum = _per_type(ranked.filter(pl.col("rk") <= n).group_by("org_type").agg(pl.col("val").sum()), "val")
    top3: dict[str, list] = {}
    for r in ranked.filter(pl.col("rk") <= 3).sort("rk").iter_rows(named=True):
        if r["val"]:  # zero-value entries carry no information for the share pie
            top3.setdefault(r["org_type"], []).append((r["author"], int(r["val"])))
    return topn_sum, top3


def _size_stats(df: pl.DataFrame, has_alltime: bool) -> dict[str, dict]:
    """Per org type: count and mean/SD of log10(rows) from size_categories tags,
    unweighted plus weighted by each metric (weighted SD via E[x^2] - mean^2)."""
    s = (
        df.select(
            "org_type", "downloads", "downloadsAllTime", "likes",
            pl.col("tags").list.eval(pl.element().filter(pl.element().str.starts_with("size_categories:")))
            .list.first().alias("sc"),
        )
        .drop_nulls("sc")
        .with_columns(pl.col("sc").str.strip_prefix("size_categories:").replace_strict(SIZE_LOG10_MID, default=None).alias("x"))
        .drop_nulls("x")
    )
    return _log_stats_per_type(s, has_alltime)


def _bytes_stats(df: pl.DataFrame, has_alltime: bool) -> dict[str, dict]:
    """Same statistics on log10(mainSize) — repo size in bytes, min files from ~2026-07 only."""
    if df["mainSize"].null_count() == len(df):
        return {}
    s = (
        df.select("org_type", "downloads", "downloadsAllTime", "likes", "mainSize")
        .filter(pl.col("mainSize") > 0)
        .with_columns(pl.col("mainSize").log(10).alias("x"))
    )
    return _log_stats_per_type(s, has_alltime)


def _log_stats_per_type(s: pl.DataFrame, has_alltime: bool) -> dict[str, dict]:
    metrics = [("dl", "downloads"), ("lk", "likes")] + ([("dlat", "downloadsAllTime")] if has_alltime else [])
    aggs = [pl.len().alias("cnt"), pl.col("x").mean().alias("u_m"), pl.col("x").std(ddof=0).alias("u_s")]
    for key, col in metrics:
        w = pl.col(col).fill_null(0)
        aggs.append(((pl.col("x") * w).sum() / w.sum()).alias(f"{key}_m"))
        aggs.append(((pl.col("x") ** 2 * w).sum() / w.sum()).alias(f"{key}_x2"))
    out: dict[str, dict] = {}
    for r in s.group_by("org_type").agg(aggs).iter_rows(named=True):
        d = {"cnt": r["cnt"], "u_m": r["u_m"], "u_s": r["u_s"]}
        for key, _ in metrics:
            mean, x2 = r[f"{key}_m"], r[f"{key}_x2"]
            ok = mean is not None and math.isfinite(mean)
            d[f"{key}_m"] = mean if ok else None
            d[f"{key}_s"] = max(0.0, x2 - mean * mean) ** 0.5 if ok else None
        out[r["org_type"]] = d
    return out


def scan_week(df: pl.DataFrame, seen: pl.DataFrame | None, top_pool: int, top_n: int) -> tuple[dict, pl.DataFrame]:
    """Pass 1 for one week: per-type vectors, concentration sums, author pool.

    Returns (week dict, updated seen-ids frame). `seen` accumulates every _id
    ever observed so `new` counts first appearances (notebook cell 24 semantics).
    """
    has_alltime = df["downloadsAllTime"].null_count() < len(df)

    g = df.group_by("org_type").agg(
        pl.col("downloads").sum().alias("dl"),
        pl.col("downloadsAllTime").sum().alias("dlat"),
        pl.col("likes").sum().alias("lk"),
        pl.len().alias("n"),
    )
    new = df.select("_id", "org_type") if seen is None else df.join(seen, on="_id", how="anti")
    seen = df.select("_id") if seen is None else pl.concat([seen, new.select("_id")])

    week = {
        "sig": (len(df), int(df["likes"].sum() or 0)),
        "dl": _per_type(g, "dl"),
        "dlat": _per_type(g, "dlat") if has_alltime else [None] * len(ORG_TYPES),
        "lk": _per_type(g, "lk"),
        "n": _per_type(g, "n"),
        "new": _per_type(new.group_by("org_type").agg(pl.len().alias("cnt")), "cnt"),
    }
    conc_metrics = ["downloads", "likes"] + (["downloadsAllTime"] if has_alltime else [])
    short = {"downloads": "dl", "likes": "lk", "downloadsAllTime": "dlat"}
    for m in conc_metrics:
        week[f"c_ds_{short[m]}"] = _topn_per_type(df, m, top_n)
        week[f"c_org_{short[m]}"], week[f"top3_{short[m]}"] = _org_level_per_type(df, m, top_n)
    if not has_alltime:
        week["c_ds_dlat"] = week["c_org_dlat"] = [None] * len(ORG_TYPES)
        week["top3_dlat"] = {}
    _, week["top3_n"] = _org_level_per_type(df, None, top_n)
    week["size"] = _size_stats(df, has_alltime)
    week["size_b"] = _bytes_stats(df, has_alltime)

    by_author = df.group_by("author").agg(
        pl.col("downloads").sum().alias("dl"),
        pl.col("downloadsAllTime").sum().alias("dlat"),
        pl.col("likes").sum().alias("lk"),
        pl.len().alias("n"),
    )
    pool = set()
    for m in ("dl", "lk", "n") + (("dlat",) if has_alltime else ()):
        pool |= set(by_author.top_k(min(top_pool, len(by_author)), by=m)["author"])
    week["pool"] = pool
    return week, seen


def detect_frozen(entity: str, dates: list[str], weeks: list[dict], full_build: bool) -> list[int]:
    """Week indices whose (row count, likes sum) signature equals the prior week's."""
    frozen = [i for i in range(1, len(weeks)) if weeks[i]["sig"] == weeks[i - 1]["sig"]]
    frozen_dates = [dates[i] for i in frozen]
    log.info("Frozen weeks detected: %s", frozen_dates or "none")
    if full_build and entity in KNOWN_FROZEN and frozen_dates != KNOWN_FROZEN[entity]:
        log.warning("Frozen weeks differ from the known list %s — upstream data may have changed", KNOWN_FROZEN[entity])
    return frozen


def round_sig(v: int | None, sig: int = 3) -> int | None:
    """Round to `sig` significant figures (series values only, for JSON compactness)."""
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


def collect_org_series(snapshots: list[tuple[str, Path]], selected: set[str], orgs: pl.DataFrame) -> dict[str, dict]:
    """Pass 2: weekly dl/lk/n series per pooled author, plus its last-seen baked type."""
    n_weeks = len(snapshots)
    sel = pl.Series("author", sorted(selected)).implode()
    records: dict[str, dict] = {}
    for week, (date_str, path) in enumerate(snapshots):
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("author").is_in(sel))
            .group_by("author")
            .agg(
                pl.col("downloads").sum().alias("dl"),
                pl.col("downloadsAllTime").sum().alias("dlat"),
                pl.col("likes").sum().alias("lk"),
                pl.len().alias("n"),
                pl.col("org_type").drop_nulls().last().alias("baked_type"),
            )
            .collect()
        )
        for r in df.iter_rows(named=True):
            rec = records.setdefault(r["author"], {"dl": [None] * n_weeks, "dlat": [None] * n_weeks, "lk": [None] * n_weeks, "n": [None] * n_weeks})
            rec["dl"][week] = round_sig(int(r["dl"] or 0))
            rec["dlat"][week] = round_sig(int(r["dlat"])) if r["dlat"] is not None else None
            rec["lk"][week] = round_sig(int(r["lk"] or 0))
            rec["n"][week] = int(r["n"])
            if r["baked_type"]:
                rec["baked_type"] = r["baked_type"]  # overwritten every appearance -> latest wins
        log.info("[%s] pass 2: matched %d of %d pooled authors", date_str, len(df), len(selected))
    return records


def build_entity(entity: str, args) -> dict | None:
    snapshots = list_snapshots(entity)
    if not snapshots:
        log.warning("No snapshot files for entity %r in %s — emitting null", entity, DATA_DIRS[entity])
        return None
    if args.quick:
        snapshots = snapshots[:: args.quick] + ([snapshots[-1]] if snapshots[-1] not in snapshots[:: args.quick] else [])
        log.warning("--quick %d: weekly deltas become multi-week aggregates; frozen-week check skipped", args.quick)
    dates = [d for d, _ in snapshots]
    log.info("[%s] building from %d weeks (%s … %s)", entity, len(dates), dates[0], dates[-1])

    orgs = load_orgs(args.orgs_csv)

    weeks, selected, seen = [], set(), None
    for date_str, path in snapshots:
        df = read_week(path, orgs)
        week, seen = scan_week(df, seen, args.top_pool, args.top_n)
        selected |= week.pop("pool")
        weeks.append(week)
        log.info("[%s] pass 1: n=%d pooled authors so far=%d", date_str, week["sig"][0], len(selected))

    frozen = detect_frozen(entity, dates, weeks, full_build=not args.quick)

    records = collect_org_series(snapshots, selected, orgs)

    # by_type[key][type_idx][week] — kept exact (tiny) so notebook numbers match 1:1
    by_type_keys = ["dl", "dlat", "lk", "n", "new", "c_ds_dl", "c_ds_lk", "c_ds_dlat", "c_org_dl", "c_org_lk", "c_org_dlat"]
    by_type = {k: [[w[k][t] for w in weeks] for t in range(len(ORG_TYPES))] for k in by_type_keys}

    # top3[metric][type_idx][week] = [[name_idx, value] x <=3] | null — for the distribution pie
    top3_names: list[str] = []
    name_idx: dict[str, int] = {}
    top3 = {m: [[None] * len(weeks) for _ in ORG_TYPES] for m in ("dl", "lk", "dlat", "n")}
    for wi, w in enumerate(weeks):
        for m in ("dl", "lk", "dlat", "n"):
            for ti, t in enumerate(ORG_TYPES):
                entries = w[f"top3_{m}"].get(t)
                if not entries:
                    continue
                packed = []
                for author, v in entries:
                    if author not in name_idx:
                        name_idx[author] = len(top3_names)
                        top3_names.append(author)
                    packed.append([name_idx[author], round_sig(v)])
                top3[m][ti][wi] = packed

    # size[stat][type_idx][week] — log10-rows stats per type, rounded to 2 decimals
    size_keys = ["cnt", "u_m", "u_s", "dl_m", "dl_s", "lk_m", "lk_s", "dlat_m", "dlat_s"]

    def size_val(w: dict, source: str, key: str, t: str):
        d = w[source].get(t)
        v = d.get(key) if d else None
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            return None
        return v if key == "cnt" else round(v, 2)

    size = {k: [[size_val(w, "size", k, t) for w in weeks] for t in ORG_TYPES] for k in size_keys}
    size_b = {k: [[size_val(w, "size_b", k, t) for w in weeks] for t in ORG_TYPES] for k in size_keys}

    org_meta = {r["author"]: (r["csv_type"], r["followers"]) for r in orgs.filter(pl.col("author").is_in(sorted(selected))).iter_rows(named=True)}
    type_index = {t: i for i, t in enumerate(ORG_TYPES)}
    org_rows = []
    for author in sorted(records, key=str.lower):
        rec = records[author]
        csv_type, followers = org_meta.get(author, (None, None))
        org_type = csv_type or rec.get("baked_type") or "individual"
        if org_type not in type_index:
            org_type = "individual"
        dl_start, dl_vals = trim_series(rec["dl"])
        dlat_start, dlat_vals = trim_series(rec["dlat"])
        lk_start, lk_vals = trim_series(rec["lk"])
        n_start, n_vals = trim_series(rec["n"])
        org_rows.append([author, type_index[org_type], followers,
                         dl_start, dl_vals, lk_start, lk_vals, n_start, n_vals, dlat_start, dlat_vals])

    return {
        "label": ENTITY_LABELS.get(entity, entity),
        "dates": dates,
        "alltime_start": next((i for i, d in enumerate(dates) if d >= ALLTIME_START), len(dates)),
        "frozen": frozen,
        "top_n": args.top_n,
        "by_type": by_type,
        "top3": top3,
        "top3_names": top3_names,
        "size": size,
        "size_b": size_b,
        "orgs": org_rows,
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
    ap.add_argument("--out", type=Path, default=OUT_HTML, help="Output HTML path.")
    ap.add_argument("--top-pool", type=int, default=TOP_POOL, help="Per-week per-metric author selection pool.")
    ap.add_argument("--top-n", type=int, default=TOP_N, help="Top-N for the concentration overlay.")
    ap.add_argument("--orgs-csv", type=Path, default=None, help="Org scrape CSV (default: newest match).")
    ap.add_argument("--entities", default="datasets", help="Comma list of entities to build (datasets[,models]).")
    ap.add_argument("--quick", type=int, default=None, metavar="N", help="Use every Nth week (always incl. latest).")
    args = ap.parse_args()

    if args.orgs_csv is None:
        args.orgs_csv = newest_orgs_csv()
    if args.orgs_csv is None or not args.orgs_csv.exists():
        log.error("No orgs CSV found (looked for %s)", ORGS_GLOB)
        return 1

    entities: dict[str, dict | None] = {}
    for entity in [e.strip() for e in args.entities.split(",") if e.strip()]:
        if entity not in DATA_DIRS:
            log.warning("Unknown entity %r — emitting null", entity)
            entities[entity] = None
            continue
        entities[entity] = build_entity(entity, args)
    if "models" not in entities:
        entities["models"] = None  # keeps the UI toggle present (disabled) until models are ingested

    if not any(entities.values()):
        log.error("No entity produced data — nothing to write")
        return 1

    payload = {
        "built": date.today().isoformat(),
        "schema": 1,
        "org_types": ORG_TYPES,
        "orgs_scrape": args.orgs_csv.name,
        "entities": entities,
    }
    for name, e in entities.items():
        if e:
            log.info("Payload[%s]: %d weeks, %d orgs", name, len(e["dates"]), len(e["orgs"]))
    render_html(payload, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
