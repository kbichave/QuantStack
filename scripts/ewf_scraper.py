#!/usr/bin/env python3
"""
EWF Scraper — Elliott Wave Forecast member chart downloader.

Fetches charts and reports from member.elliottwave-forecast.com (Level 3 / Group 3).

URL patterns (confirmed 2026-04-04):
  Per-instrument:  /chart-details/group-3/{ticker.lower()}
  Blue Box:        /blue-box-report/group-3
  Market Overview: /group-page/group-3/daily-market-overview-3
  Weekly Overview: /group-page/group-3/weekly-market-overview-3

Tab pane IDs (global, same for every instrument — confirmed 2026-04-04):
  1h_premarket  #instrument-update-type-href-10
  1h_midday     #instrument-update-type-href-21
  4h            #instrument-update-type-href-14  (default active tab)
  daily         #instrument-update-type-href-17
  weekly        #instrument-update-type-href-20
  (post-market  #instrument-update-type-href-7   — locked, not in subscription)

Charts are served as static PNG/JPG from /storage/instrument-charts/{Month Year}/{hash}.{ext}
— downloaded directly, not screenshotted.

Blue Box Report lists instruments + Bullish/Bearish with direct storage image links.

Output:
  data/ewf/{YYYY-MM-DD}/{update_type}/{instrument}.png
  data/ewf/{YYYY-MM-DD}/{update_type}/metadata.json
  data/ewf/.session/state.json      (persistent auth cookie)
  data/ewf/.session/debug/          (debug command output)

Commands:
  python scripts/ewf_scraper.py login           # force re-login, save session
  python scripts/ewf_scraper.py debug           # screenshot dashboard + dump nav
  python scripts/ewf_scraper.py blue_box        # fetch Blue Box report
  python scripts/ewf_scraper.py market_overview # fetch daily market overview
  python scripts/ewf_scraper.py 1h_premarket    # fetch 1H pre-market charts
  python scripts/ewf_scraper.py 1h_midday       # fetch 1H midday charts
  python scripts/ewf_scraper.py 4h              # fetch 4H charts
  python scripts/ewf_scraper.py daily           # fetch daily charts
  python scripts/ewf_scraper.py weekly          # fetch weekly overview

Env:
  EWF_USERNAME    member username
  EWF_PASSWORD    member password
  EWF_HEADLESS    set "false" to show browser window (debugging)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from playwright.async_api import async_playwright, BrowserContext, Page

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://member.elliottwave-forecast.com"
LOGIN_URL = f"{BASE_URL}/login"

CHART_BASE = f"{BASE_URL}/chart-details/group-3"
BLUE_BOX_URL = f"{BASE_URL}/blue-box-report/group-3"
MARKET_OVERVIEW_URL = f"{BASE_URL}/group-page/group-3/daily-market-overview-3"
WEEKLY_OVERVIEW_URL = f"{BASE_URL}/group-page/group-3/weekly-market-overview-3"

SESSION_DIR = Path("data/ewf/.session")
DEBUG_DIR = SESSION_DIR / "debug"
OUTPUT_DIR = Path("data/ewf")
STATE_FILE = SESSION_DIR / "state.json"

STOCKS = [
    "AAL", "AAPL", "AMD", "AMZN", "BA", "BABA", "BAC",
    "GOOGL", "META", "MSFT", "NFLX", "NKE", "NVDA", "TSLA", "XOM",
]
ETFS = [
    "GDX", "IWM", "QQQ", "SPY", "XLE", "XLF", "XLI", "XLP", "XLV", "XLY", "XME",
]
ALL_INSTRUMENTS = STOCKS + ETFS

# Global tab pane IDs — confirmed consistent across all instruments 2026-04-04.
# Post-Market (#7) is excluded (not in Level 3 subscription).
TIMEFRAME_PANE = {
    "1h_premarket": "#instrument-update-type-href-10",
    "1h_midday":    "#instrument-update-type-href-21",
    "4h":           "#instrument-update-type-href-14",
    "daily":        "#instrument-update-type-href-17",
    "weekly":       "#instrument-update-type-href-20",
}

# Update types that fetch per-instrument chart pages
INSTRUMENT_UPDATE_TYPES = set(TIMEFRAME_PANE.keys())

ALL_UPDATE_TYPES = INSTRUMENT_UPDATE_TYPES | {"blue_box", "market_overview"}


def instrument_url(ticker: str) -> str:
    return f"{CHART_BASE}/{ticker.lower()}"


# ---------------------------------------------------------------------------
# Browser context
# ---------------------------------------------------------------------------


def _headless() -> bool:
    return os.environ.get("EWF_HEADLESS", "true").lower() != "false"


async def _build_context(pw) -> BrowserContext:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    browser = await pw.chromium.launch(headless=_headless())
    storage = str(STATE_FILE) if STATE_FILE.exists() else None
    context = await browser.new_context(
        storage_state=storage,
        viewport={"width": 1920, "height": 1080},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    )
    return context


async def _is_authenticated(context: BrowserContext) -> bool:
    page = await context.new_page()
    try:
        await page.goto(f"{BASE_URL}/dashboard", wait_until="networkidle", timeout=15000)
        return "login" not in page.url.lower()
    except Exception:
        return False
    finally:
        await page.close()


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------


async def login(context: BrowserContext) -> bool:
    username = os.environ.get("EWF_USERNAME", "").strip()
    password = os.environ.get("EWF_PASSWORD", "").strip()
    if not username or not password:
        raise EnvironmentError("EWF_USERNAME and EWF_PASSWORD must be set in .env")

    page = await context.new_page()
    try:
        await page.goto(LOGIN_URL, wait_until="networkidle", timeout=20000)
        await page.fill("#username", username)
        await page.fill("#password", password)

        try:
            async with page.expect_navigation(wait_until="networkidle", timeout=20000):
                await page.click('#frmLogin button[type="submit"]')
        except Exception:
            await page.wait_for_load_state("networkidle", timeout=10000)

        if "/login" in page.url:
            try:
                err_el = await page.wait_for_selector(
                    '.alert, .alert-danger, [class*="error"]', timeout=3000
                )
                err = (await err_el.inner_text()).strip() if err_el else "unknown"
            except Exception:
                err = "unknown"
            print(f"[EWF] Login FAILED — {err}")
            return False

        await context.storage_state(path=str(STATE_FILE))
        print(f"[EWF] Logged in — session saved")
        return True

    except Exception as exc:
        print(f"[EWF] Login exception: {exc}")
        return False
    finally:
        await page.close()


async def ensure_authenticated(context: BrowserContext) -> bool:
    if await _is_authenticated(context):
        return True
    print("[EWF] Session expired — re-authenticating...")
    return await login(context)


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------


async def _download_image(page: Page, img_url: str, out_path: Path) -> bool:
    """Download a storage image using the authenticated browser session."""
    try:
        resp = await page.request.get(img_url)
        if resp.status != 200:
            return False
        out_path.write_bytes(await resp.body())
        return True
    except Exception as exc:
        print(f"[EWF] Download failed {img_url}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Per-instrument chart fetcher
# ---------------------------------------------------------------------------


async def _get_chart_url(page: Page, pane_id: str) -> str | None:
    """
    From the current chart-details page, get the image URL for the given tab pane.
    Returns the /storage/... URL or None if the pane has no chart (locked or missing).
    """
    img_el = await page.query_selector(
        f'{pane_id} img.chart__image, {pane_id} img[src*="/storage/"]'
    )
    if not img_el:
        return None
    src = await img_el.get_attribute("src")
    return src if src and "/storage/" in src else None


async def fetch_instrument_charts(
    context: BrowserContext, update_type: str, date_str: str
) -> None:
    """
    Navigate to each instrument's chart-details page, pull the image URL from
    the correct tab pane, and download it directly.
    """
    pane_id = TIMEFRAME_PANE[update_type]
    out_dir = OUTPUT_DIR / date_str / update_type
    out_dir.mkdir(parents=True, exist_ok=True)

    page = await context.new_page()
    results: dict[str, str] = {}

    try:
        for instrument in ALL_INSTRUMENTS:
            url = instrument_url(instrument)
            try:
                await page.goto(url, wait_until="networkidle", timeout=20000)
                if "login" in page.url.lower():
                    print(f"[EWF] Session expired mid-run — stopping")
                    break

                img_url = await _get_chart_url(page, pane_id)
                if not img_url:
                    print(f"[EWF]   - {instrument}: no chart in pane {pane_id}")
                    results[instrument] = "no_chart"
                    continue

                out_path = out_dir / f"{instrument}.png"
                ok = await _download_image(page, img_url, out_path)
                results[instrument] = "ok" if ok else "download_failed"
                print(f"[EWF]   {'✓' if ok else '✗'} {instrument}  {img_url.split('/')[-1]}")

            except Exception as exc:
                print(f"[EWF]   ✗ {instrument}: {exc}")
                results[instrument] = "error"
    finally:
        await page.close()

    _save_metadata(out_dir, update_type, date_str, results)
    ok_count = sum(1 for v in results.values() if v == "ok")
    print(f"[EWF] {update_type}: {ok_count}/{len(ALL_INSTRUMENTS)} downloaded → {out_dir}")


# ---------------------------------------------------------------------------
# Blue Box report
# ---------------------------------------------------------------------------


async def fetch_blue_box_report(context: BrowserContext, date_str: str) -> None:
    """
    Parse the Blue Box Report page for instrument → chart image URL mappings,
    download each chart image directly.

    Page structure (confirmed 2026-04-04):
      - Each row has two <a href="/storage/..."> tags: one with ticker text, one "View Chart"
      - Only instruments with active blue box setups appear (varies daily, typically 3-10)
    """
    out_dir = OUTPUT_DIR / date_str / "blue_box"
    out_dir.mkdir(parents=True, exist_ok=True)

    page = await context.new_page()
    try:
        await page.goto(BLUE_BOX_URL, wait_until="networkidle", timeout=20000)
        if "login" in page.url.lower():
            print("[EWF] Blue Box: session expired")
            return

        # Each chart link appears twice — once with the ticker as text, once as "View Chart".
        # Collect unique href→ticker pairs.
        raw_links: list[dict] = await page.evaluate("""() =>
            Array.from(document.querySelectorAll('a[href*="/storage/instrument-charts/"]'))
                .map(a => ({text: a.textContent.trim(), href: a.href}))
        """)

        # Build ticker → image URL map (use the link whose text is a known instrument)
        image_map: dict[str, str] = {}
        for link in raw_links:
            if link["text"].upper() in ALL_INSTRUMENTS:
                image_map[link["text"].upper()] = link["href"]

        print(f"[EWF] Blue Box: {len(image_map)} instruments have setups today: {list(image_map.keys())}")

        results: dict[str, str] = {}
        for instrument, img_url in image_map.items():
            out_path = out_dir / f"{instrument}.png"
            ok = await _download_image(page, img_url, out_path)
            results[instrument] = "ok" if ok else "download_failed"
            print(f"[EWF]   {'✓' if ok else '✗'} {instrument}  {img_url.split('/')[-1]}")

        # Also grab direction (Bullish/Bearish) from page text
        directions: dict[str, str] = {}
        main_text = await page.inner_text("main")
        for line in main_text.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].upper() in ALL_INSTRUMENTS:
                directions[parts[0].upper()] = parts[1] if parts[1] in ("Bullish", "Bearish") else "unknown"

        results["directions"] = directions  # type: ignore[assignment]
        _save_metadata(out_dir, "blue_box", date_str, results)
        print(f"[EWF] Blue Box: {len(image_map)} charts saved → {out_dir}")

    except Exception as exc:
        print(f"[EWF] Blue Box failed: {exc}")
    finally:
        await page.close()


# ---------------------------------------------------------------------------
# Overview pages (market overview + weekly)
# ---------------------------------------------------------------------------


async def fetch_overview_page(
    context: BrowserContext, update_type: str, date_str: str
) -> None:
    """Full-page screenshot of the daily or weekly market overview page."""
    url = WEEKLY_OVERVIEW_URL if update_type == "weekly" else MARKET_OVERVIEW_URL
    out_dir = OUTPUT_DIR / date_str / update_type
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{update_type}_{date_str}.png"

    page = await context.new_page()
    try:
        await page.goto(url, wait_until="networkidle", timeout=20000)
        if "login" in page.url.lower():
            print(f"[EWF] {update_type}: session expired")
            return
        await page.screenshot(path=str(out_path), full_page=True)
        _save_metadata(out_dir, update_type, date_str, {"screenshot": "ok", "url": url})
        print(f"[EWF] {update_type} → {out_path}")
    except Exception as exc:
        print(f"[EWF] {update_type} failed: {exc}")
    finally:
        await page.close()


# ---------------------------------------------------------------------------
# Debug
# ---------------------------------------------------------------------------


async def run_debug(context: BrowserContext) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    page = await context.new_page()
    try:
        await page.goto(f"{BASE_URL}/dashboard", wait_until="networkidle", timeout=20000)
        shot = DEBUG_DIR / "dashboard.png"
        await page.screenshot(path=str(shot), full_page=True)

        links: list[dict] = await page.evaluate("""() => {
            const seen = new Set();
            return Array.from(document.querySelectorAll('a[href]'))
                .filter(a => a.href.includes('member.elliottwave') && !seen.has(a.href) && seen.add(a.href))
                .map(a => ({text: (a.textContent || '').trim().slice(0, 60), href: a.href}));
        }""")

        links_file = DEBUG_DIR / "all_links.json"
        with open(links_file, "w") as f:
            json.dump(links, f, indent=2)

        print(f"[EWF:debug] {len(links)} links → {links_file}")
        print(f"[EWF:debug] Dashboard screenshot → {shot}")
        for link in links:
            print(f"  {link['text'][:45]:45s} {link['href']}")

        # Verify SPY chart page and show available tabs
        await page.goto(instrument_url("SPY"), wait_until="networkidle", timeout=15000)
        spy_shot = DEBUG_DIR / "spy_chart.png"
        await page.screenshot(path=str(spy_shot), full_page=True)
        tabs: list[dict] = await page.evaluate("""() =>
            Array.from(document.querySelectorAll('#myTab .nav-link'))
                .map(a => ({
                    label: a.textContent.trim(),
                    pane_id: a.getAttribute('href'),
                    active: a.classList.contains('active')
                }))
        """)
        print(f"\n[EWF:debug] SPY chart tabs:")
        for t in tabs:
            print(f"  {'*' if t['active'] else ' '} {t['label']:25s} {t['pane_id']}")
        print(f"[EWF:debug] SPY screenshot → {spy_shot}")

    finally:
        await page.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_metadata(out_dir: Path, update_type: str, date_str: str, results: dict) -> None:
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(
            {
                "update_type": update_type,
                "date": date_str,
                "fetched_at_utc": datetime.utcnow().isoformat(),
                "results": results,
            },
            f, indent=2,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(cmd: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")

    async with async_playwright() as pw:
        context = await _build_context(pw)

        if cmd == "login":
            STATE_FILE.unlink(missing_ok=True)
            sys.exit(0 if await login(context) else 1)

        if not await ensure_authenticated(context):
            print("[EWF] Authentication failed — check EWF_USERNAME / EWF_PASSWORD in .env")
            sys.exit(1)

        if cmd == "debug":
            await run_debug(context)
        elif cmd == "blue_box":
            await fetch_blue_box_report(context, date_str)
        elif cmd in ("market_overview", "weekly"):
            await fetch_overview_page(context, cmd, date_str)
        elif cmd in INSTRUMENT_UPDATE_TYPES:
            await fetch_instrument_charts(context, cmd, date_str)
        else:
            print(f"[EWF] Unknown command: {cmd!r}")
            print(f"      Valid: {sorted(ALL_UPDATE_TYPES | {'debug', 'login'})}")
            sys.exit(1)

        await context.browser.close()


def main() -> None:
    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else "1h_premarket"))


if __name__ == "__main__":
    main()
