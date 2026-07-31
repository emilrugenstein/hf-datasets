"""Build the dataset -> model adoption index from a weekly *models* snapshot.

"Model adoption" for a dataset = how many models declare it as training data.
On the hub that declaration surfaces as a `dataset:<dataset_id>` entry in the
model's `tags` (derived by HF from the model card's `datasets:` field), so the
index is just a count of distinct models carrying each such tag.

alex-repo/datasets_with_model_usage.parquet holds the same measure for
2026-07-15 only, with no script to regenerate it. This rebuilds it from any
snapshot date, defaulting to the latest, so the adoption metric can be computed
on the same week as the dataset snapshot instead of carried forward.

The 1.3 GB source is streamed in row-group batches (only `tags` is read) and the
cached blob is deleted afterwards, matching hf_datasets_snapshot_batch.py.

Usage:
  python hf_models_dataset_usage.py                 # latest snapshot
  python hf_models_dataset_usage.py --date 2026-07-15
  python hf_models_dataset_usage.py --keep-cache
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, scan_cache_dir

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")

REPO_ID = "hfmlsoc/hub_weekly_snapshots"
OUT_DIR = Path("data/hf_models_dataset_usage")
SNAPSHOT_RE = re.compile(r"^models/(\d{4}-\d{2}-\d{2})/models\.parquet$")
BATCH_ROWS = 100_000
TAG_PREFIX = "dataset:"

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
log = logging.getLogger("hf_models_dataset_usage")
logging.getLogger("huggingface_hub.utils._cache_manager").setLevel(logging.ERROR)


def latest_date(api: HfApi) -> str:
    files = api.list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
    dates = sorted({m.group(1) for f in files if (m := SNAPSHOT_RE.match(f))})
    if not dates:
        raise RuntimeError(f"no model snapshots found in {REPO_ID}")
    return dates[-1]


def count_adoption(path: str) -> tuple[Counter, int, int]:
    """Distinct models declaring each dataset, plus (models seen, models with >=1 link)."""
    counts: Counter = Counter()
    seen = linked = 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_ROWS, columns=["tags"]):
        for tags in batch.column("tags").to_pylist():
            seen += 1
            if not tags:
                continue
            # set(): a model card can repeat a dataset, but it adopts it once
            ds = {t[len(TAG_PREFIX):] for t in tags if t and t.startswith(TAG_PREFIX)}
            if ds:
                linked += 1
                counts.update(ds)
        if seen % 500_000 == 0:
            log.info("  %s models scanned, %s datasets seen", f"{seen:,}", f"{len(counts):,}")
    return counts, seen, linked


def delete_cached(filename: str) -> None:
    """Drop the snapshot blob from the HF cache so peak disk stays bounded."""
    try:
        cache = scan_cache_dir()
    except Exception as e:
        log.warning("scan_cache_dir failed: %s, leaving cache as-is", e)
        return
    revs = [rev.commit_hash for repo in cache.repos if repo.repo_id == REPO_ID
            for rev in repo.revisions
            if any(f.file_path.as_posix().endswith(filename) for f in rev.files)]
    if revs:
        try:
            cache.delete_revisions(*revs).execute()
        except Exception as e:
            log.warning("delete_revisions failed: %s, cache will grow", e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None, help="Snapshot date (YYYY-MM-DD). Default: latest.")
    ap.add_argument("--keep-cache", action="store_true", help="Do not delete the downloaded blob.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = HfApi(token=HF_TOKEN)
    date = args.date or latest_date(api)
    filename = f"models/{date}/models.parquet"
    log.info("Snapshot %s (auth: %s)", date, "token" if HF_TOKEN else "anonymous")

    try:
        path = hf_hub_download(REPO_ID, repo_type="dataset", filename=filename, token=HF_TOKEN)
        counts, seen, linked = count_adoption(path)
    finally:
        if not args.keep_cache:
            delete_cached(filename)

    out = (pd.DataFrame({"dataset_id": list(counts), "n_models": list(counts.values())})
             .sort_values("n_models", ascending=False, ignore_index=True))
    out_path = OUT_DIR / f"dataset_model_usage_{date}.parquet"
    out.to_parquet(out_path, index=False)

    log.info("%s models scanned, %s declared >=1 dataset (%.1f%%)", f"{seen:,}", f"{linked:,}", 100 * linked / seen)
    log.info("%s distinct datasets adopted, %s links total", f"{len(out):,}", f"{out.n_models.sum():,}")
    log.info("wrote %s", out_path)
    print(out.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
