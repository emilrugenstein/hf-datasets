"""Build the AI World "impact per unit of presence" viz (outcome indices by org type).

Redoes the indexed-impact analysis from alex-repo/owner_type_charts.py, but on
this repo's scraped `org_type` classification instead of a hand-made owner list,
and with a scope switch (all datasets / top 1,000 by downloads).

Each metric is an INDEX: the type's share of an outcome divided by its share of
datasets, so 1.0x means "exactly proportional to how many datasets it publishes".
Alex's chart plotted both forms of the adoption metric as if they were two
findings; they are algebraically identical, so only one appears here.

Three metrics are emitted: downloads, likes and model adoption. Government and
classroom fall to n=1 and n=4 in the top-1,000 scope, far too few to index; the
viz drops them from that scope rather than drawing them.

Two scopes, one fixed x domain, so switching shows how much of any apparent
"impact" difference is really an artefact of selecting on downloads: inside the
top 1,000 every dataset is already a download outlier, so the indices collapse
toward 1.0.

Individual accounts are excluded throughout (`org_type` is null for them, and
folding nulls into "individual" would make it a residual bucket for every
unclassifiable account). The top-1,000 scope is the 1,000 most-downloaded
ORG-OWNED datasets, matching viewer/final-aiw-viz/dataset_size_by_org_type.html.

Model adoption comes from the models snapshot of the SAME week, built by
hf_models_dataset_usage.py. It previously had to be carried forward from
alex-repo's 2026-07-15 index; recomputing on 2026-07-22 moved no index by more
than 0.01x, and the two agree on 98.5% of datasets exactly (r = 0.9999).

Writes the DATA block into viewer/final-aiw-viz/dataset_impact_index_by_org_type.html
between the __DATA_START__/__DATA_END__ sentinels, leaving the file self-contained.

Usage:
  python hf_datasets_impact_index_build_viz.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SNAPSHOT = "2026-07-22"
SNAP = Path(f"data/hf_datasets_historic/hf_datasets_min_{SNAPSHOT}.parquet")
USAGE = Path(f"data/hf_models_dataset_usage/dataset_model_usage_{SNAPSHOT}.parquet")
OUT = Path("viewer/final-aiw-viz/dataset_impact_index_by_org_type.html")

TOP_N = 1000
# Ordered so the emitted rows are stable; the viz declares its own display order.
TYPES = ["company", "university", "community", "non-profit", "classroom", "government"]


def load() -> pd.DataFrame:
    """Snapshot restricted to org-owned datasets, with model adoption joined on."""
    df = pd.read_parquet(SNAP, columns=["id", "author", "likes", "downloadsAllTime", "org_type"])
    usage = pd.read_parquet(USAGE).rename(columns={"dataset_id": "id", "n_models": "n_models_using"})
    df = df.merge(usage, on="id", how="left")
    # left join: a dataset no model declares simply has no row in the index
    df["n_models_using"] = df["n_models_using"].fillna(0)
    return df[df["org_type"].notna()]


def indices(scope: pd.DataFrame, pop: pd.Series) -> list[dict]:
    """Per org type: the five outcome indices plus the raw values behind them."""
    g = scope.groupby("org_type")
    agg = pd.DataFrame({
        "n": g.size(),
        "downloads": g["downloadsAllTime"].sum(),
        "likes": g["likes"].sum(),
        "adoption": g["n_models_using"].sum(),
    }).reindex(TYPES)
    share = agg["n"] / agg["n"].sum()

    rows = []
    for t in TYPES:
        if pd.isna(agg.loc[t, "n"]):
            continue
        a = agg.loc[t]
        rows.append({
            "type": t,
            "n": int(a["n"]),
            "pop": int(pop[t]),
            # share-based indices: outcome share / dataset share
            "dl": a["downloads"] / agg["downloads"].sum() / share[t],
            "lk": a["likes"] / agg["likes"].sum() / share[t],
            "ad": a["adoption"] / agg["adoption"].sum() / share[t],
            # raw values, for the tooltip
            "downloads": int(a["downloads"]),
            "likes": int(a["likes"]),
            "adoption": int(a["adoption"]),
        })
    return rows


def fmt(rows: list[dict]) -> str:
    """One aligned JS object literal per (type x scope) row."""
    out = []
    for r in rows:
        out.append(
            '  { type:"%s", scope:"%s", n:%d, pop:%d, '
            'dl:%.4f, lk:%.4f, ad:%.4f, '
            'downloads:%d, likes:%d, adoption:%d },'
            % (r["type"], r["scope"], r["n"], r["pop"], r["dl"], r["lk"], r["ad"],
               r["downloads"], r["likes"], r["adoption"])
        )
    return "\n".join(out)


def main() -> None:
    df = load()
    pop = df.groupby("org_type").size()

    rows = []
    for key, scope in [("all", df), ("top", df.nlargest(TOP_N, "downloadsAllTime"))]:
        for r in indices(scope, pop):
            r["scope"] = key
            rows.append(r)

    block = "const DATA = [\n" + fmt(rows) + "\n];"
    html = OUT.read_text(encoding="utf-8")
    patched = re.sub(
        r"(/\*__DATA_START__\*/).*?(/\*__DATA_END__\*/)",
        lambda m: m.group(1) + "\n" + block + "\n" + m.group(2),
        html, flags=re.S,
    )
    OUT.write_text(patched, encoding="utf-8")

    print(f"{len(df):,} org-owned datasets; wrote {len(rows)} rows to {OUT}")
    for r in rows:
        print(f"  {r['scope']:>3} {r['type']:<11} n={r['n']:>6}  dl={r['dl']:.2f}x  lk={r['lk']:.2f}x  "
              f"ad={r['ad']:.2f}x")


if __name__ == "__main__":
    main()
