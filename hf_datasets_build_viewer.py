"""Build the interactive HTML trend viewer from the weekly snapshot minima.

Two passes over `data/hf_datasets_historic/hf_datasets_min_<date>.parquet`:
  pass 1 — per-week hub totals, per-topic aggregates (task_categories tags +
           top free-form tags), and per-week top-1000 `_id` sets per metric;
  pass 2 — weekly download series for the union of "ever in a top-1000"
           datasets (this makes client-side top-N ranking exact for any week).

The compact JSON payload replaces the /*__DATA__*/ placeholder in
`viewer/template.html` and is written to `viewer/hf_datasets_viewer.html` —
a single self-contained file, no server needed.

Usage:
  python hf_datasets_build_viewer.py             # full build (~2-4 min)
  python hf_datasets_build_viewer.py --quick 8   # every 8th week, fast iteration
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

DATA_DIR = Path("data/hf_datasets_historic")
TEMPLATE = Path("viewer/template.html")
OUT_HTML = Path("viewer/hf_datasets_viewer.html")

ALLTIME_START = "2025-02-26"  # downloadsAllTime is null before this snapshot
TOP_POOL = 1000               # per-week per-metric pool for dataset selection
TOP_FREEFORM = 60             # free-form tags kept in the topic vocabulary
TASK_PREFIX = "task_categories:"

FILE_RE = re.compile(r"hf_datasets_min_(\d{4}-\d{2}-\d{2})\.parquet$")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf_datasets_build_viewer")

# Maps a raw tag to its topic name: task_categories are stripped to the task,
# unprefixed community tags pass through, all other namespaces become null.
TOPIC_EXPR = (
    pl.when(pl.col("tags").str.starts_with(TASK_PREFIX))
    .then(pl.col("tags").str.strip_prefix(TASK_PREFIX))
    .when(~pl.col("tags").str.contains(":", literal=True))
    .then(pl.col("tags"))
    .otherwise(None)
    .alias("topic")
)

PAPER_EXPR = (
    pl.col("tags")
    .list.eval(pl.element().str.starts_with("arxiv:"))
    .list.any()
    .alias("has_paper")
)


def list_snapshots() -> list[tuple[str, Path]]:
    """Sorted (date, path) pairs; the date list is the viewer's shared x-axis."""
    pairs = [(m.group(1), p) for p in DATA_DIR.glob("hf_datasets_min_*.parquet") if (m := FILE_RE.search(p.name))]
    return sorted(pairs)


def build_topic_vocab(latest_path: Path) -> list[dict]:
    """Topic vocabulary from the latest snapshot: all task_categories + top free-form tags."""
    counts = (
        pl.scan_parquet(latest_path)
        .select("_id", "tags")
        .explode("tags", empty_as_null=True)
        .with_columns(TOPIC_EXPR, is_task=pl.col("tags").str.starts_with(TASK_PREFIX))
        .drop_nulls("topic")
        .unique(subset=["_id", "topic"])
        .group_by("topic")
        .agg(pl.len().alias("n"), pl.col("is_task").any())
        .collect()
    )
    tasks = counts.filter(pl.col("is_task")).sort("n", descending=True)
    task_names = set(tasks["topic"])
    freeform = (
        counts.filter(~pl.col("is_task") & ~pl.col("topic").is_in(task_names))
        .sort("n", descending=True)
        .head(TOP_FREEFORM)
    )
    vocab = [{"name": t, "kind": "task"} for t in tasks["topic"]]
    vocab += [{"name": t, "kind": "community"} for t in freeform["topic"]]
    log.info("Topic vocab: %d task_categories + %d free-form = %d topics", len(tasks), len(freeform), len(vocab))
    return vocab


def scan_week(path: Path, vocab_names: list[str], top_pool: int) -> dict:
    """Pass 1 for one week: hub totals, per-topic aggregates, top-k id sets."""
    df = pl.read_parquet(path, columns=["_id", "tags", "likes", "downloads", "downloadsAllTime"])
    has_alltime = df["downloadsAllTime"].null_count() < len(df)
    df = df.with_columns(PAPER_EXPR)
    papers = df.filter(pl.col("has_paper"))

    hub = {
        "n": len(df),
        "dl": int(df["downloads"].sum()),
        "dlat": int(df["downloadsAllTime"].sum()) if has_alltime else None,
        "lk": int(df["likes"].sum() or 0),
        "n_paper": len(papers),
        "dl_paper": int(papers["downloads"].sum()),
        "dlat_paper": int(papers["downloadsAllTime"].sum()) if has_alltime else None,
        "lk_paper": int(papers["likes"].sum() or 0),
    }

    topic_rows = (
        df.select("_id", "tags", "likes", "downloads", "downloadsAllTime")
        .explode("tags", empty_as_null=True)
        .with_columns(TOPIC_EXPR)
        .drop_nulls("topic")
        .filter(pl.col("topic").is_in(vocab_names))
        .unique(subset=["_id", "topic"])
        .group_by("topic")
        .agg(
            pl.len().alias("n"),
            pl.col("downloads").sum().alias("dl"),
            pl.col("downloadsAllTime").sum().alias("dlat"),
            pl.col("likes").sum().alias("lk"),
        )
    )
    topics = {
        r["topic"]: (r["n"], int(r["dl"]), int(r["dlat"]) if has_alltime else None, int(r["lk"] or 0))
        for r in topic_rows.iter_rows(named=True)
    }

    top_ids = set(df.top_k(top_pool, by="downloads")["_id"])
    top_ids |= set(df.drop_nulls("likes").top_k(top_pool, by="likes")["_id"])
    if has_alltime:
        top_ids |= set(df.drop_nulls("downloadsAllTime").top_k(top_pool, by="downloadsAllTime")["_id"])

    # (_id, downloadsAllTime) of this week, for the cross-week growth selection in main()
    grow = df.select("_id", "downloadsAllTime").drop_nulls() if has_alltime else None
    return {"hub": hub, "topics": topics, "top_ids": top_ids, "grow": grow}


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


def collect_series(snapshots: list[tuple[str, Path]], selected: set[str]) -> dict[str, dict]:
    """Pass 2: weekly series + latest-appearance metadata for every selected dataset."""
    n_weeks = len(snapshots)
    ids = pl.Series("_id", sorted(selected)).implode()
    records: dict[str, dict] = {}
    for week, (date_str, path) in enumerate(snapshots):
        df = (
            pl.scan_parquet(path)
            .filter(pl.col("_id").is_in(ids))
            .select("_id", "id", "author", "createdAt", "tags", "likes", "downloads", "downloadsAllTime", "org_type")
            .collect()
        )
        for r in df.iter_rows(named=True):
            rec = records.setdefault(r["_id"], {"dl": [None] * n_weeks, "dlat": [None] * n_weeks, "lk": [None] * n_weeks})
            rec["dl"][week] = round_sig(r["downloads"])
            rec["dlat"][week] = round_sig(r["downloadsAllTime"])
            rec["lk"][week] = round_sig(r["likes"])
            # overwritten every appearance → latest wins (handles renames)
            rec["meta"] = {
                "id": r["id"],
                "org_type": r["org_type"] or "individual",
                # createdAt is a datetime in newer snapshots but a raw string in the oldest
                "createdAt": str(r["createdAt"])[:10] if r["createdAt"] else None,
                "tags": r["tags"],
            }
        log.info("[%s] pass 2: matched %d of %d selected", date_str, len(df), len(selected))
    return records


def topics_of(tags: list[str], topic_index: dict[str, int]) -> list[int]:
    idxs = set()
    for t in tags:
        name = t[len(TASK_PREFIX):] if t.startswith(TASK_PREFIX) else (t if ":" not in t else None)
        if name in topic_index:
            idxs.add(topic_index[name])
    return sorted(idxs)


def build_payload(snapshots, vocab, weeks, records) -> dict:
    dates = [d for d, _ in snapshots]
    alltime_start = next((i for i, d in enumerate(dates) if d >= ALLTIME_START), len(dates))

    hub = {k: [w["hub"][k] for w in weeks] for k in ("n", "dl", "dlat", "lk", "n_paper", "dl_paper", "dlat_paper", "lk_paper")}

    topics = []
    for entry in vocab:
        name = entry["name"]
        per_week = [w["topics"].get(name, (0, 0, None, 0)) for w in weeks]
        topics.append({
            "name": name,
            "kind": entry["kind"],
            "n": [t[0] for t in per_week],
            "dl": [t[1] for t in per_week],
            "dlat": [t[2] for t in per_week],
            "lk": [t[3] for t in per_week],
        })
    topic_index = {e["name"]: i for i, e in enumerate(vocab)}

    org_types = sorted({rec["meta"]["org_type"] for rec in records.values()})
    org_index = {o: i for i, o in enumerate(org_types)}

    datasets = []
    for rec in records.values():
        m = rec["meta"]
        dl_start, dl_vals = trim_series(rec["dl"])
        dlat_start, dlat_vals = trim_series(rec["dlat"])
        lk_start, lk_vals = trim_series(rec["lk"])
        arxiv = [t[len("arxiv:"):] for t in m["tags"] if t.startswith("arxiv:")]
        datasets.append([
            m["id"], org_index[m["org_type"]], m["createdAt"],
            topics_of(m["tags"], topic_index), arxiv,
            dl_start, dl_vals, dlat_start, dlat_vals, lk_start, lk_vals,
        ])
    datasets.sort(key=lambda d: d[0].lower())

    return {
        "built": date.today().isoformat(),
        "dates": dates,
        "alltime_start": alltime_start,
        "hub": hub,
        "topics": topics,
        "org_types": org_types,
        "datasets": datasets,
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
    ap.add_argument("--top-pool", type=int, default=TOP_POOL, help="Per-week per-metric selection pool.")
    ap.add_argument("--quick", type=int, default=None, metavar="N", help="Use every Nth week (always incl. latest).")
    args = ap.parse_args()

    snapshots = list_snapshots()
    if not snapshots:
        log.error("No snapshot files in %s", DATA_DIR)
        return 1
    if args.quick:
        snapshots = snapshots[:: args.quick] + ([snapshots[-1]] if snapshots[-1] not in snapshots[:: args.quick] else [])
    log.info("Building from %d weeks (%s … %s)", len(snapshots), snapshots[0][0], snapshots[-1][0])

    vocab = build_topic_vocab(snapshots[-1][1])
    vocab_names = [e["name"] for e in vocab]

    weeks, selected = [], set()
    baseline = None  # (_id, base) — first-seen downloadsAllTime, for growth ranking
    for date_str, path in snapshots:
        w = scan_week(path, vocab_names, args.top_pool)
        selected |= w["top_ids"]
        if w["grow"] is not None:
            g = w["grow"]
            new = g.rename({"downloadsAllTime": "base"}) if baseline is None else (
                g.join(baseline, on="_id", how="anti").rename({"downloadsAllTime": "base"}))
            baseline = new if baseline is None else pl.concat([baseline, new])
            growth = g.join(baseline, on="_id").with_columns((pl.col("downloadsAllTime") - pl.col("base")).alias("g"))
            selected |= set(growth.top_k(args.top_pool, by="g")["_id"])
        del w["grow"]
        weeks.append(w)
        log.info("[%s] pass 1: n=%d selected so far=%d", date_str, w["hub"]["n"], len(selected))

    records = collect_series(snapshots, selected)
    payload = build_payload(snapshots, vocab, weeks, records)
    log.info("Payload: %d datasets, %d topics, %d weeks", len(payload["datasets"]), len(payload["topics"]), len(payload["dates"]))

    render_html(payload, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
