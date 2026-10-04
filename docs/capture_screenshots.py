"""Capture UI screenshots for the README.

Kept in the repository so the images are reproducible rather than hand-cropped. Requires
the API and the frontend dev server to be running:

    ./.venv/bin/python -m uvicorn backend.app:app --port 8000
    cd frontend && npm run dev

Then:

    ./.venv/bin/python docs/capture_screenshots.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://localhost:5173"
OUT = Path("docs/images")
VIEWPORT = {"width": 1480, "height": 1000}

# Each shot names a route and the part of it worth showing. `tab` selects a tab before
# capture; `anchor` scrolls a matching element to centre, which survives content edits
# that a hard pixel offset would not.
SHOTS = [
    {"name": "dashboard", "route": "/", "offset": 0, "settle": 2.6},
    {"name": "dashboard-run", "route": "/", "offset": 1180, "settle": 2.2},
    {"name": "landscape", "route": "/landscape", "offset": 120, "settle": 2.6},
    {"name": "qubo", "route": "/qubo", "offset": 980, "settle": 2.4},
    {"name": "quantum", "route": "/quantum", "offset": 60, "settle": 2.6},
    {"name": "benchmark", "route": "/benchmark", "offset": 1150, "settle": 2.6},
    {"name": "results", "route": "/results", "offset": 0, "settle": 2.6},
    {"name": "research", "route": "/research", "offset": 0, "settle": 2.4,
     "tab": "Mutation extrapolation"},
    {"name": "research-math", "route": "/research", "offset": 0, "settle": 2.4,
     "tab": "Literature review", "anchor": ".katex-display-wrap", "anchor_index": 2},
]

# Captured at 2x then re-encoded to WebP. The raw 2x PNGs are ~4 MB each; WebP keeps the
# retina crispness at roughly a twentieth of the size, which is the difference between a
# 31 MB repo and a 1 MB one.
WEBP_QUALITY = 82


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=2)

        try:
            page.goto(BASE, timeout=20_000)
        except Exception as exc:
            print(f"cannot reach {BASE}: {exc}", file=sys.stderr)
            print("start the frontend with: cd frontend && npm run dev", file=sys.stderr)
            browser.close()
            return 1

        for shot in SHOTS:
            name, route = shot["name"], shot["route"]
            # `networkidle` is unreliable here: the API keeps a health poll open, so the
            # page is usable long before the network goes quiet.
            page.goto(f"{BASE}{route}", wait_until="domcontentloaded", timeout=30_000)
            if tab := shot.get("tab"):
                try:
                    page.get_by_role("button", name=tab, exact=True).click(timeout=30_000)
                except Exception:
                    print(f"  ! {name}: tab {tab!r} not found", file=sys.stderr)
            if anchor := shot.get("anchor"):
                page.wait_for_selector(anchor, timeout=30_000)
                page.evaluate(
                    "([sel, idx]) => { const e = document.querySelectorAll(sel)[idx];"
                    " if (e) e.scrollIntoView({ block: 'center' }); }",
                    [anchor, shot.get("anchor_index", 0)],
                )
            elif shot["offset"]:
                page.evaluate(f"window.scrollTo(0, {shot['offset']})")
            # Reveal animations are scroll-triggered, so settle after scrolling, not before.
            time.sleep(shot["settle"])
            png = OUT / f"{name}.png"
            page.screenshot(path=str(png))

            from PIL import Image
            webp = OUT / f"{name}.webp"
            Image.open(png).convert("RGB").save(webp, "WEBP",
                                                quality=WEBP_QUALITY, method=6)
            png.unlink()
            print(f"  wrote {webp}  ({webp.stat().st_size / 1024:.0f} KB)")

        browser.close()
    print(f"\n{len(SHOTS)} screenshots in {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
