"""Build the AI World-styled ranking viewer (top accounts + account statistics).

Same two-pass pipeline as `hf_org_trends_build_viewer.py` (imported, not copied);
this script only swaps the template, prunes the payload, and writes its own output.
The dashboard is rendered from `viewer/org_template_aiworld.html`, which follows
the AI World (aiworld.eu / CEPS) house style — D3 v7, system-ui typography,
selector-pill controls, and a source + AI World logo footer. Note: unlike the base
viewer, the AI World template loads D3 from a CDN, so it needs internet to render.

The template keeps only the ranking tiles, so everything the removed tiles needed
(size statistics, the concentration/newly-added series, the top-N accounts buckets)
is dropped from the payload — `size_f` alone is the bulk of the base viewer's file
size. KEEP_* below is the contract: extend it when the template reads a new key.

Usage:
  python hf_org_trends_build_viewer_aiworld.py             # full build
  python hf_org_trends_build_viewer_aiworld.py --quick 8   # every 8th week, fast iteration
"""

from __future__ import annotations

import sys
from pathlib import Path

import hf_org_trends_build_viewer as base

base.TEMPLATE = Path("viewer/org_template_aiworld.html")
base.OUT_HTML = Path("viewer/hf_org_trends_viewer_aiworld.html")

KEEP_ENTITY = {"label", "dates", "alltime_start", "by_type", "top3", "top3_names", "orgs"}
KEEP_BY_TYPE = {"dl", "dlat", "lk", "n"}  # the pie's four rank metrics

_render_html = base.render_html


def render_html(payload: dict, out_path: Path) -> None:
    """Drop the keys the ranking template never reads, then render as usual."""
    for entity in payload["entities"].values():
        if not entity:
            continue
        for key in set(entity) - KEEP_ENTITY:
            del entity[key]
        for key in set(entity["by_type"]) - KEEP_BY_TYPE:
            del entity["by_type"][key]
    _render_html(payload, out_path)


base.render_html = render_html

if __name__ == "__main__":
    sys.exit(base.main())
