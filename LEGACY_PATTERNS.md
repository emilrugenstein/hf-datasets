# How the old (models) project worked — patterns worth reusing

Summary of the capstone thesis pipeline at
`C:\Users\mail\OneDrive\Dokumente\StudiumLUC\Capstone\Programming`, written so this
repo can reuse its patterns without digging through the old notebooks.

## Pipeline shape

1. **Retrieve** — `hf_snapshot_retrieve.ipynb` pulled a current full snapshot via the
   HF API; `hf_org_scrapper.py` scraped org metadata (type, followers, badge) from
   the HF website into `hf_orgs_scraped_<date>.csv`.
2. **Prepare** — `hf_main_prepare.ipynb`: load snapshot → clean → merge orgs →
   write a slim "min" parquet. Same steps later automated per-week in
   `hf_snapshot_batch.py` (the direct ancestor of this repo's batch script).
3. **Analyse** — one notebook per theme: `hf_main_general_trends.ipynb`
   (cross-section: org types, engagement distributions), `hf_main_historic_trends.ipynb`
   (longitudinal), `hf_main_licenses.ipynb` (license evolution).

## Data handling rules that transfer directly

- **Join snapshots on `_id`** (immutable Mongo ObjectID), never on `id`
  (`author/name` — changes on rename).
- **`downloads` is a 30-day rolling window**; `downloadsAllTime` is cumulative.
  `downloadsAllTime` + `license` (via `cardData`) only exist from **2025-02-26**
  — the notebooks used a constant `more_extensive_data_date = "2025-02-26"` to
  split eras. Same boundary confirmed for datasets snapshots.
- **Schema fallback when batch-reading old snapshots**: intersect wanted columns
  with the file's actual schema, fill the rest with None (see
  `_read_with_schema_fallback` in the batch script).
- **License extraction**: `cardData` is a JSON string; `license` can be a string
  or (rarely) a list → comma-join lists so parquet gets one dtype.
- **Orgs caveat**: `org_type` comes from a *single* scrape date applied to all
  historic snapshots — wrong for orgs that changed/renamed/didn't exist yet.
  Acceptable for descriptive stats; flag it in any org-type-over-time figure.
- **NaN `org_type` → `"individual"`** (authors without an org page).
- Atomic writes: write `.tmp` then `os.replace`, so an interrupted batch never
  leaves a corrupt min file.

## Plotting conventions (thesis-grade static figures)

- seaborn theme `whitegrid`, context `paper`, serif font, dpi 150, no legend frame.
- **One fixed color per org_type everywhere**: `ORG_TYPE_COLORS` dict built from
  `sns.color_palette("colorblind")`, keyed in `value_counts()` order — never
  recompute per chart.
- **Titles via `print("...")` before `plt.show()`**, not `suptitle`. No `savefig`.
- Engagement metrics (likes, downloads, trendingScore) are heavy-tailed with many
  zeros → **symlog y-axis (`linthresh=1`)**.
- **Pitfall**: seaborn ≥0.13 `boxplot` with `palette=` but no `hue=` rendered an
  *empty figure* on this data. Use raw `ax.boxplot(patch_artist=True)` with manual
  face colors from `ORG_TYPE_COLORS` instead.
- Monthly bar charts: use a string index (`strftime("%Y-%m")`) — pandas' bar path
  on a DatetimeIndex fails with `Must supply freq`.
- Category exclusions are **scoped per chart, not global** (e.g. pies dropped
  `classroom`/`government` for label legibility; cumulative area dropped
  `individual` because its long tail dominates) — always print a full summary
  table including excluded groups after the figure.

## What was deliberately NOT carried over

- The epoch_ai / fuzzy-matching machinery (`rapidfuzz`, manual review JSONs,
  `merge.ipynb`) — model-specific, irrelevant for datasets.
- The activity filter (`downloadsAllTime > 199` / 30-day `downloads > 49`) —
  thresholds were tuned for models; this repo keeps all rows and filters at
  analysis time (see README "Open decisions").
