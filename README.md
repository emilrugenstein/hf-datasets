# hf-datasets — statistics on Hugging Face datasets

Follow-up to the capstone thesis project (which analysed HF **models**). This repo
analyses the **datasets** side of the Hub, reusing the proven patterns from the old
project — see [LEGACY_PATTERNS.md](LEGACY_PATTERNS.md) for a summary of how that
pipeline worked and which conventions to keep.

Old project location (read-only reference):
`C:\Users\mail\OneDrive\Dokumente\StudiumLUC\Capstone\Programming`

## Data source

`hfmlsoc/hub_weekly_snapshots` (HF dataset repo) — weekly full-hub snapshots.
Verified facts (2026-07-22):

- `datasets/<YYYY-MM-DD>/datasets.parquet`, **104 weeks, 2024-07-24 → 2026-07-15**
  (repo also has `models/`, `spaces/`, `daily_papers/`).
- Rows grow 185k (earliest) → 963k (latest), unfiltered.
- Schema eras: `cardData` + `trendingScore` from ~2024-10; `downloadsAllTime` from
  **2025-02-26** (same boundary as models); `mainSize` from ~2026 H1 only.
- `downloads` is assumed to be a 30-day rolling window like for models — **not yet
  verified for datasets specifically**.

Per-column compressed sizes in the latest snapshot (377 MB total): `cardData` 204 MB,
`description` 56 MB, `sha` 39 MB — everything else together ~77 MB. `tags` is cheap
(8 MB) despite being a list column.

## Pipeline

1. `hf_datasets_snapshot_batch.py` — downloads each weekly snapshot, merges
   `org_type` from the orgs scrape, writes slim
   `data/hf_datasets_historic/hf_datasets_min_<date>.parquet` files (gitignored,
   regenerable). License is NOT a separate column — derive it from `license:*`
   entries in `tags` at analysis time. Start with `--smoke-test`.
2. `hf_datasets_explore.ipynb` — tinkering notebook; loads selected columns of the
   latest snapshot remotely (footer/column reads, no full download).
3. `hf_org_scrapper.py` + `data/hf_orgs/hf_orgs_scraped_2026-04-30.csv` — copied
   unchanged from the old project (see caveat in LEGACY_PATTERNS.md).
4. `hf_datasets_build_viewer.py` — builds the interactive trend viewer
   `viewer/hf_datasets_viewer.html` (gitignored, ~9 MB, self-contained — just
   double-click it). Aggregates all weekly min files: hub totals, per-topic trends
   (`task_categories` + top-60 free-form tags), and full weekly series for every
   dataset that was ever in a weekly top-1000 by any ranking (~10k). The UI
   (`viewer/template.html`) filters by min value, top 100/1000, topics, and papers
   attached (`arxiv:` tags), switchable between four metrics: 30-day downloads,
   all-time downloads, all-time growth (downloads gained since 2025-02-26), and
   likes. Re-run after new snapshots; `--quick 8` for fast iteration.
5. `hf_datasets_historic_trends.ipynb` — static thesis-style trend figures, ported
   from the capstone's `hf_main_historic_trends.ipynb` (models): downloads/likes/
   counts/newly-added by org type, top-N concentration, spike attribution,
   single-dataset history, and license trends. Licenses are derived from
   `license:*` entries in `tags`, so license trends cover the full 104 weeks
   (unlike models, where license only existed from 2025-02-26).

## ⚠ Open decisions

- **Columns: DECIDED** (`_id, id, author, createdAt, tags, likes, downloads,
  downloadsAllTime, trendingScore` + `org_type`); the full 104-week batch has run.
  Changing the set later means re-downloading all snapshots (raws are deleted
  after processing).
- **Paper references:** `paperswithcode_id` is NOT the only carrier — papers also
  appear as `arxiv:XXXX.XXXXX` entries inside `tags`, and `citation` holds free-text
  BibTeX. Coverage of the three was not yet quantified; check while tinkering.
- **No activity filter** (decided): the batch keeps all rows; "active dataset"
  filtering happens at analysis time. The models thresholds (`downloadsAllTime > 199`
  / 30-day `downloads > 49`) were tuned for models and may not transfer.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`.env` with `HF_TOKEN=...` was copied from the old project (gitignored). Then:

```powershell
python hf_datasets_snapshot_batch.py --dry-run
python hf_datasets_snapshot_batch.py --smoke-test
```
