"""Build the standalone AI World "Who drives the Hub" viz (stacked area).

One pass over the weekly snapshot minima, reusing the loaders from
`hf_org_trends_build_viewer` (same org-type join rules): per-week per-org-type
sums of all-time downloads and likes. Frozen upstream weeks (snapshot identical
to the prior week) are smoothed at build time: the first fresh week's increment
is spread evenly across the frozen run plus that week (linear interpolation),
so the cumulative curves show no flat-then-jump artefact.

The published file is `viewer/final-aiw-viz/dataset_downloads_by_org_type.html`,
one self-contained file in the AI World house style, ready to host at
viz.aiworld.eu (e.g. aiworld/Story/<viz-slug>/viz.html). The JSON payload
replaces /*__DATA__*/ in the template; since the template file was retired, a
rebuild recovers it from the published viz by swapping the embedded DATA back
to the placeholder.

Usage:
  python hf_org_drivers_build_viz.py             # full build
  python hf_org_drivers_build_viz.py --quick 8   # every 8th week, fast iteration
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import date
from pathlib import Path

import polars as pl

import hf_org_trends_build_viewer as base

TEMPLATE = Path("viewer/org_drivers_viz_template.html")
OUT_HTML = Path("viewer/final-aiw-viz/dataset_downloads_by_org_type.html")


def load_template() -> str:
    """Template text, recovered from the published viz when the file is absent."""
    if TEMPLATE.exists():
        return TEMPLATE.read_text(encoding="utf-8")
    html = OUT_HTML.read_text(encoding="utf-8")
    template, n = re.subn(r"const DATA = \{.*?\};", "const DATA = /*__DATA__*/;", html, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError(f"neither {TEMPLATE} nor a recoverable DATA block in {OUT_HTML}")
    return template

log = logging.getLogger("hf_org_drivers_build_viz")


def smooth_frozen(vals: list[int | None], frozen: set[int]) -> list[int | None]:
    """Linear-interpolate each frozen run: the next fresh week's increment is
    spread evenly over the run plus that week. Runs without a non-null anchor
    on both sides (e.g. trailing, or before all-time tracking) stay untouched."""
    out = list(vals)
    i = 1
    while i < len(out):
        if i not in frozen:
            i += 1
            continue
        j = i
        while j < len(out) and j in frozen:
            j += 1
        if j < len(out) and out[i - 1] is not None and out[j] is not None:
            anchor, delta, span = out[i - 1], out[j] - out[i - 1], j - i + 1
            for k in range(i, j):
                out[k] = round(anchor + delta * (k - i + 1) / span)
        i = j + 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_HTML, help="Output HTML path.")
    ap.add_argument("--orgs-csv", type=Path, default=None, help="Org scrape CSV (default: newest match).")
    ap.add_argument("--quick", type=int, default=None, metavar="N", help="Use every Nth week (always incl. latest).")
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
    if args.quick:
        snapshots = snapshots[:: args.quick] + ([snapshots[-1]] if snapshots[-1] not in snapshots[:: args.quick] else [])
        log.warning("--quick %d: sampled weeks — frozen-week detection/smoothing is unreliable", args.quick)
    dates = [d for d, _ in snapshots]
    log.info("Building from %d weeks (%s … %s)", len(dates), dates[0], dates[-1])

    orgs = base.load_orgs(args.orgs_csv)
    types = base.ORG_TYPES
    dlat: list[list[int | None]] = [[] for _ in types]
    lk: list[list[int | None]] = [[] for _ in types]
    sigs: list[tuple[int, int]] = []
    for date_str, path in snapshots:
        df = base.read_week(path, orgs)
        has_alltime = df["downloadsAllTime"].null_count() < len(df)
        g = df.group_by("org_type").agg(
            pl.col("downloadsAllTime").sum().alias("dlat"),
            pl.col("likes").sum().alias("lk"),
        )
        by = {r["org_type"]: r for r in g.iter_rows(named=True)}
        for ti, t in enumerate(types):
            r = by.get(t)
            dlat[ti].append(int(r["dlat"]) if has_alltime and r and r["dlat"] is not None else None)
            lk[ti].append(int(r["lk"] or 0) if r else 0)
        sigs.append((len(df), int(df["likes"].sum() or 0)))
        log.info("[%s] n=%d", date_str, len(df))

    frozen = {i for i in range(1, len(sigs)) if sigs[i] == sigs[i - 1]}
    frozen_dates = sorted(dates[i] for i in frozen)
    log.info("Frozen weeks smoothed: %s", frozen_dates or "none")
    if not args.quick and frozen_dates != base.KNOWN_FROZEN["datasets"]:
        log.warning("Frozen weeks differ from the known list %s — upstream data may have changed", base.KNOWN_FROZEN["datasets"])
    dlat = [smooth_frozen(s, frozen) for s in dlat]
    lk = [smooth_frozen(s, frozen) for s in lk]

    payload = {
        "built": date.today().isoformat(),
        "dates": dates,
        "org_types": types,
        "dlat": dlat,
        "lk": lk,
    }
    template = load_template()
    html = template.replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")), 1)
    tmp = args.out.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    os.replace(tmp, args.out)
    log.info("Wrote %s (%.1f KB)", args.out, len(html) / 1e3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
