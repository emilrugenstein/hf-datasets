"""Build the AI World-styled org-type trends viewer.

Same two-pass pipeline and JSON payload as `hf_org_trends_build_viewer.py`
(imported, not copied); only the template and default output differ:
the dashboard is rendered from `viewer/org_template_aiworld.html`, which follows
the AI World (aiworld.eu / CEPS) house style — D3 v7, system-ui typography,
selector-pill controls, and a source + AI World logo footer. Note: unlike the
base viewer, the AI World template loads D3 from a CDN, so it needs internet
access to render.

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

if __name__ == "__main__":
    sys.exit(base.main())
