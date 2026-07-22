"""Crawl https://huggingface.co/organizations and dump org metadata to CSV.

Output columns: slug, org_type, model_count, followers, badge.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

BASE_URL = "https://huggingface.co/organizations"
OUTPUT_DIR = Path("data/hf_orgs")
DEFAULT_OUTPUT_CSV = OUTPUT_DIR / f"hf_orgs_scraped_{date.today().isoformat()}.csv"
CHECKPOINT_PATH = OUTPUT_DIR / "_checkpoint.parquet"

REQUEST_TIMEOUT = 30
MAX_RETRIES = 5
DEFAULT_DELAY = 1.5  # seconds between requests; actual sleep is delay * uniform(0.75, 1.25)
USER_AGENT = "hf-org-scrapper/1.0 (research; contact emil.rugenstein@ceps.eu)"

PAGE_FALLBACK_TOTAL = 8000

RATE_FACTOR_MAX = 8.0
RATE_FACTOR_DECAY = 0.6
RATE_FACTOR_MIN_AFTER_429 = 8.0

_delay_factor = 1.0

MODELS_RE = re.compile(r"([\d.,]+\s*[kKmM]?)\s*models?\b", flags=re.IGNORECASE)
FOLLOWERS_RE = re.compile(r"([\d.,]+\s*[kKmM]?)\s*followers?\b", flags=re.IGNORECASE)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hf_org_scrapper")


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"})

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if token:
        s.headers["Authorization"] = f"Bearer {token}"

    cookie = os.environ.get("HF_SESSION_COOKIE")
    if cookie:
        s.cookies.set("token", cookie, domain="huggingface.co")
        log.info("Using HF session cookie (logged-in browser session)")
    elif token:
        log.info("Using HF access token (Bearer; may be ignored on website routes)")
    else:
        log.info("No HF auth — running anonymous")
    return s


def parse_count(text: str) -> int:
    """Parse strings like '128k', '1.09k', '4.66M', '93.9k', '55' into int."""
    text = text.strip().replace(",", "")
    m = re.match(r"([\d.]+)\s*([kKmM]?)", text)
    if not m:
        return 0
    num = float(m.group(1))
    suffix = m.group(2).lower()
    mult = {"": 1, "k": 1000, "m": 1000000}[suffix]
    return int(round(num * mult))


def parse_card(a_tag) -> dict | None:
    href = a_tag.get("href", "")
    if not href.startswith("/") or "/" in href[1:]:
        return None
    text = a_tag.get_text(" ", strip=True)
    f_match = FOLLOWERS_RE.search(text)
    if not f_match:
        return None

    type_span = a_tag.find("span", class_="capitalize")
    org_type = type_span.get_text(" ", strip=True).lower() if type_span else None

    badge_span = a_tag.find("span", class_=lambda c: c and "-skew-x-12" in c)
    badge = badge_span.get_text(" ", strip=True) if badge_span else None

    m_match = MODELS_RE.search(text)
    model_count = parse_count(m_match.group(1)) if m_match else 0

    return {
        "slug": href.lstrip("/"),
        "org_type": org_type,
        "model_count": model_count,
        "followers": parse_count(f_match.group(1)),
        "badge": badge,
    }


def get_total_pages(session: requests.Session) -> int:
    try:
        resp = session.get(BASE_URL, params={"p": 0}, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError as e:
        log.warning("Could not fetch listing page (%s); falling back to %d pages", e, PAGE_FALLBACK_TOTAL)
        return PAGE_FALLBACK_TOTAL
    soup = BeautifulSoup(resp.text, "html.parser")
    max_p = 0
    for a in soup.find_all("a", href=True):
        m = re.search(r"[?&]p=(\d+)", a["href"])
        if m:
            max_p = max(max_p, int(m.group(1)))
    if max_p == 0:
        log.warning("Could not parse pagination; falling back to %d pages", PAGE_FALLBACK_TOTAL)
        return PAGE_FALLBACK_TOTAL
    return max_p + 1


def fetch_page(session: requests.Session, page_num: int) -> list[dict]:
    global _delay_factor
    last_status: int | None = None
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(BASE_URL, params={"p": page_num}, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                rows: list[dict] = []
                for a in soup.find_all("a", href=True):
                    parsed = parse_card(a)
                    if parsed:
                        rows.append(parsed)
                _delay_factor = max(_delay_factor * RATE_FACTOR_DECAY, 1.0)
                return rows
            last_status = resp.status_code
            if resp.status_code == 429:
                _delay_factor = min(max(_delay_factor * 2, RATE_FACTOR_MIN_AFTER_429), RATE_FACTOR_MAX)
                retry_after_hdr = resp.headers.get("Retry-After", "") or ""
                try:
                    retry_after = int(retry_after_hdr.strip())
                except ValueError:
                    retry_after = 0
                cooldown = retry_after if retry_after > 0 else 30
                log.warning(
                    "page %d rate-limited (429); cooling down %ds, backoff %.1fx, will retry on next run",
                    page_num, cooldown, _delay_factor,
                )
                time.sleep(cooldown)
                raise RuntimeError(f"page {page_num} rate-limited (429)")
            if resp.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"page {page_num} failed: status={last_status} exc={last_exc}")


def load_checkpoint() -> tuple[pd.DataFrame, set[int]]:
    if not CHECKPOINT_PATH.exists():
        return pd.DataFrame(columns=["slug", "org_type", "model_count", "followers", "badge", "_page"]), set()
    df = pd.read_parquet(CHECKPOINT_PATH)
    done = set(df["_page"].astype(int).unique().tolist())
    log.info("Resuming from checkpoint: %d pages already scraped (%d rows)", len(done), len(df))
    return df, done


def save_checkpoint(df: pd.DataFrame) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CHECKPOINT_PATH, index=False)


def crawl(start: int, end: int | None, delay: float) -> pd.DataFrame:
    session = make_session()
    if end is None:
        end = get_total_pages(session)
        log.info("Total pages detected: %d", end)

    accumulated, done_pages = load_checkpoint()
    pages_to_do = [p for p in range(start, end) if p not in done_pages]
    if not pages_to_do:
        log.info("Nothing to do — all %d pages already in checkpoint", end - start)
        return accumulated

    new_rows: list[dict] = []
    failed: list[int] = []
    completed_since_save = 0

    try:
        for p in tqdm(pages_to_do, desc="pages"):
            try:
                rows = fetch_page(session, p)
                for r in rows:
                    r["_page"] = p
                new_rows.extend(rows)
            except Exception as e:
                failed.append(p)
                log.warning("page %d failed: %s", p, e)
            completed_since_save += 1
            if completed_since_save >= 100:
                accumulated = pd.concat([accumulated, pd.DataFrame(new_rows)], ignore_index=True)
                save_checkpoint(accumulated)
                new_rows = []
                completed_since_save = 0
            time.sleep(delay * random.uniform(0.75, 1.25) * _delay_factor)
    except KeyboardInterrupt:
        log.warning("Interrupted — flushing checkpoint")
        accumulated = pd.concat([accumulated, pd.DataFrame(new_rows)], ignore_index=True)
        save_checkpoint(accumulated)
        raise

    if new_rows:
        accumulated = pd.concat([accumulated, pd.DataFrame(new_rows)], ignore_index=True)
        save_checkpoint(accumulated)

    if failed:
        log.warning("%d pages skipped (rate-limited or error) — re-run to retry them: %s", len(failed), failed[:20])

    return accumulated


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None, help="Exclusive. Defaults to auto-detected total.")
    ap.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="Seconds between requests.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = crawl(args.start, args.end, args.delay)

    if df.empty:
        log.error("No rows scraped — aborting before writing CSV")
        return 1

    df = df.drop(columns=["_page"], errors="ignore").drop_duplicates(subset=["slug"], keep="first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    log.info("Wrote %d rows to %s", len(df), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
