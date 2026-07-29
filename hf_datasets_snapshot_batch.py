"""Batch-process every weekly *datasets* snapshot from `hfmlsoc/hub_weekly_snapshots`.

Adapted from the capstone's `hf_snapshot_batch.py` (models). For each
`datasets/<YYYY-MM-DD>/datasets.parquet` in the dataset repo, downloads to the
HF cache, extracts `license` from `cardData`, merges `org_type` from the orgs
scrape, and writes a slim per-snapshot file to
`data/hf_datasets_historic/hf_datasets_min_<date>.parquet`. The cached raw is
deleted after each snapshot so peak disk stays bounded.

DELIBERATE DIFFERENCES vs the models pipeline:
  - NO activity filter — all rows are kept (~963k in the latest snapshot).
    Filtering happens at analysis time instead; see README "Open decisions".
  - Column selection below is PROVISIONAL — final choice deferred until after
    tinkering with the latest snapshot (see README "Open decisions").

Schema eras (verified via remote parquet footers, 2026-07-22):
  - `cardData` (license source) and `trendingScore` exist only from ~2024-10.
  - `downloadsAllTime` exists only from 2025-02-26 (same boundary as models).
  - `mainSize` exists only from ~2026 H1.
  Missing columns are filled with None by the schema fallback.

CAVEAT — orgs file:
Org metadata comes from a single scrape (`hf_orgs_scraped_2026-04-30.csv`)
applied to every snapshot. For orgs that changed type, were renamed, or did
not yet exist on the snapshot date, `org_type` is inaccurate.

Usage:
  python hf_datasets_snapshot_batch.py --smoke-test   # earliest, middle, latest
  python hf_datasets_snapshot_batch.py                # full batch
  python hf_datasets_snapshot_batch.py --since 2025-01-01 --until 2025-06-30
  python hf_datasets_snapshot_batch.py --force        # reprocess existing dates
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

REPO_ID = "hfmlsoc/hub_weekly_snapshots"
OUT_DIR = Path("data/hf_datasets_historic")
ORGS_CSV = Path("data/hf_orgs/hf_orgs_scraped_2026-04-30.csv")

# PROVISIONAL column selection (decision deferred — see README "Open decisions").
# Currently: analytical core + governance/provenance. Always dropped: `cardData`
# (only its derived `license` is kept), `sha`, `key`, `description`.
# `mainSize` (repo size in bytes, ~99.9% coverage) exists only from ~2026 H1 —
# older snapshots get None via the schema fallback, and min files written
# before it was added lack the column entirely (re-run with --force to add it).
READ_COLS = [
    "_id", "id", "author", "createdAt", "tags",
    "likes", "downloads", "downloadsAllTime", "trendingScore", "mainSize"
]
KEEP_COLS = [c for c in READ_COLS if c != "cardData"] + ["org_type"]

SNAPSHOT_RE = re.compile(r"^datasets/(\d{4}-\d{2}-\d{2})/datasets\.parquet$")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf_datasets_snapshot_batch")

# huggingface_hub's _cache_manager prints noisy "Couldn't delete blob" warnings
# on Windows (no symlinks) when its rmtree races with already-cleaned paths.
logging.getLogger("huggingface_hub.utils._cache_manager").setLevel(logging.ERROR)


def list_snapshot_dates(api: HfApi) -> list[str]:
    files = api.list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
    dates = sorted({m.group(1) for f in files if (m := SNAPSHOT_RE.match(f))})
    return dates


def load_orgs(path: Path) -> pd.DataFrame:
    orgs = pd.read_csv(path)
    orgs = orgs.rename(columns={"slug": "org_slug"})
    return orgs[["org_slug", "org_type"]]


def extract_license(card) -> str | None:
    if isinstance(card, str):
        try:
            card = json.loads(card)
        except Exception:
            return None
    if not isinstance(card, dict):
        return None
    lic = card.get("license")
    if lic is None or isinstance(lic, str):
        return lic
    if isinstance(lic, list):
        # rare list-valued licenses — parquet needs one dtype, so comma-join
        return ",".join(str(x) for x in lic) if lic else None
    return str(lic)


def _read_with_schema_fallback(parquet_path: str | os.PathLike, date_str: str) -> pd.DataFrame:
    """Read READ_COLS, intersecting with whatever the file actually has.

    Older snapshots lack `cardData`, `trendingScore`, `downloadsAllTime`, and
    `mainSize` (see schema eras in the module docstring). Missing columns are
    filled with None so the merge and final projection still work.
    """
    available = set(pq.ParquetFile(parquet_path).schema_arrow.names)
    present = [c for c in READ_COLS if c in available]
    missing = [c for c in READ_COLS if c not in available]
    df = pd.read_parquet(parquet_path, columns=present)

    for c in missing:
        df[c] = None
    if missing:
        log.info("[%s] schema fallback: filled %s with None", date_str, missing)

    return df[READ_COLS]


def _delete_cached_revision(filename: str) -> None:
    """Drop this snapshot's blob from the HF cache to keep peak disk bounded."""
    try:
        cache_info = scan_cache_dir()
    except Exception as e:
        log.warning("scan_cache_dir failed: %s — leaving cache as-is", e)
        return

    revisions_to_delete = []
    for repo in cache_info.repos:
        if repo.repo_id != REPO_ID:
            continue
        for rev in repo.revisions:
            if any(f.file_path.as_posix().endswith(filename) for f in rev.files):
                revisions_to_delete.append(rev.commit_hash)

    if not revisions_to_delete:
        return
    try:
        cache_info.delete_revisions(*revisions_to_delete).execute()
    except Exception as e:
        log.warning("delete_revisions failed: %s — cache will grow", e)


def process_snapshot(date_str: str, orgs: pd.DataFrame, *, force: bool) -> str:
    """Returns 'written' | 'skipped' | 'failed'."""
    out_path = OUT_DIR / f"hf_datasets_min_{date_str}.parquet"
    if out_path.exists() and not force:
        log.info("[%s] skipped (already exists)", date_str)
        return "skipped"

    filename = f"datasets/{date_str}/datasets.parquet"
    try:
        cache_path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=filename, token=HF_TOKEN)
        df = _read_with_schema_fallback(cache_path, date_str)

        # No activity filter — deliberate, see module docstring.
        # df["license"] = df["cardData"].apply(extract_license)
        # df = df.drop(columns=["cardData"])
        df = df.merge(orgs, left_on="author", right_on="org_slug", how="left")
        df = df[KEEP_COLS]

        tmp_path = out_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, out_path)

        log.info("[%s] wrote %d rows → %s", date_str, len(df), out_path)
    except Exception as e:
        log.warning("[%s] failed: %s", date_str, e)
        return "failed"
    finally:
        _delete_cached_revision(filename)

    return "written"


def select_smoke_test_dates(dates: list[str]) -> list[str]:
    if len(dates) < 3:
        return dates
    return [dates[0], dates[len(dates) // 2], dates[-1]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Reprocess dates even if the min file exists.")
    ap.add_argument("--since", type=str, default=None, help="Earliest date to process (YYYY-MM-DD, inclusive).")
    ap.add_argument("--until", type=str, default=None, help="Latest date to process (YYYY-MM-DD, inclusive).")
    ap.add_argument("--limit", type=int, default=None, help="Process at most N dates.")
    ap.add_argument("--smoke-test", action="store_true", help="Process only earliest, middle, and latest snapshot.")
    ap.add_argument("--dry-run", action="store_true", help="List dates that would be processed and exit.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ORGS_CSV.exists():
        log.error("Orgs file not found: %s", ORGS_CSV)
        return 1
    orgs = load_orgs(ORGS_CSV)
    log.info("Loaded %d orgs from %s", len(orgs), ORGS_CSV)

    api = HfApi(token=HF_TOKEN)
    log.info("HF auth: %s", "token" if HF_TOKEN else "anonymous")
    all_dates = list_snapshot_dates(api)
    log.info("Discovered %d snapshot dates in %s (%s … %s)", len(all_dates), REPO_ID, all_dates[0], all_dates[-1])

    if args.smoke_test:
        dates = select_smoke_test_dates(all_dates)
        log.info("Smoke-test mode: %s", dates)
    else:
        dates = all_dates
        if args.since:
            dates = [d for d in dates if d >= args.since]
        if args.until:
            dates = [d for d in dates if d <= args.until]
        if args.limit:
            dates = dates[: args.limit]

    if args.dry_run:
        log.info("Dry run — would process %d dates: %s", len(dates), dates)
        return 0

    counts = {"written": 0, "skipped": 0, "failed": 0}
    failed_dates: list[str] = []
    for d in dates:
        status = process_snapshot(d, orgs, force=args.force)
        counts[status] += 1
        if status == "failed":
            failed_dates.append(d)

    log.info("Done. written=%d skipped=%d failed=%d", counts["written"], counts["skipped"], counts["failed"])
    if failed_dates:
        log.warning("Failed dates (re-run to retry): %s", failed_dates)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
