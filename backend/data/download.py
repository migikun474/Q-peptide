"""Download the public datasets Q-Peptide trains on.

Sources and licences:
  DRAMP   http://dramp.cpu-bioinfor.org/   CC BY 4.0, bulk download, no registration.
          The primary source: sequences with MIC values and hemolysis annotations.
  UniProt https://rest.uniprot.org         public REST API, no key. Optional background
          set; see research/literature_review.md section 2.2 for why random background
          negatives are treated as a limitation rather than a clean negative set.

Nothing is redistributed in the repository: `data/raw/` is gitignored and this script
reproduces it. Run:  python -m backend.data.download
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")

DRAMP_BASE = (
    "http://dramp.cpu-bioinfor.org/downloads/download.php"
    "?filename=download_data/DRAMP3.0_new/"
)
DRAMP_FILES = {
    "general_amps.xlsx": "all general AMP entries with MIC and hemolysis annotation",
    "Antibacterial_amps.xlsx": "antibacterial subset (optional)",
}

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"


def download_dramp(which: list[str] | None = None, force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    names = which or ["general_amps.xlsx"]
    out = []
    for name in names:
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"  {name}: already present ({dest.stat().st_size:,} bytes), skipping")
            out.append(dest)
            continue
        url = DRAMP_BASE + name
        print(f"  downloading {name} …", flush=True)
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        if len(r.content) < 10_000:
            raise RuntimeError(
                f"{name}: response was only {len(r.content)} bytes; the DRAMP download "
                "endpoint may have changed. Check http://dramp.cpu-bioinfor.org/downloads/"
            )
        dest.write_bytes(r.content)
        print(f"  {name}: {len(r.content):,} bytes -> {dest}")
        out.append(dest)
    return out


def download_uniprot_background(
    n: int = 2000, min_len: int = 10, max_len: int = 60, force: bool = False
) -> Path:
    """Reviewed sequences WITHOUT the antimicrobial keyword, as a background set.

    Keyword KW-0929 is UniProt's "Antimicrobial"; its negation gives non-AMP-annotated
    entries. Absence of an annotation is not evidence of absence of activity, which is
    exactly why this set is labelled "background" and not "negative".
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / "uniprot_background.json"
    if dest.exists() and not force:
        print(f"  uniprot background: already present, skipping")
        return dest

    query = (
        f"(reviewed:true) AND (length:[{min_len} TO {max_len}]) "
        "NOT (keyword:KW-0929)"
    )
    collected: list[dict] = []
    cursor = None
    print("  querying UniProt …", flush=True)
    while len(collected) < n:
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,protein_name,organism_name,length,sequence",
            "size": str(min(500, n - len(collected))),
        }
        if cursor:
            params["cursor"] = cursor
        r = requests.get(UNIPROT_URL, params=params, timeout=120)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        if not results:
            break
        for e in results:
            seq = e.get("sequence", {}).get("value")
            if seq:
                collected.append({
                    "accession": e.get("primaryAccession"),
                    "sequence": seq,
                    "length": e.get("sequence", {}).get("length"),
                    "organism": e.get("organism", {}).get("scientificName"),
                })
        link = r.headers.get("Link", "")
        cursor = None
        if 'rel="next"' in link and "cursor=" in link:
            cursor = link.split("cursor=")[1].split("&")[0].split(">")[0]
        if not cursor:
            break
        time.sleep(0.2)

    dest.write_text(json.dumps({
        "source": "UniProtKB REST API",
        "query": query,
        "licence": "UniProt data is available under CC BY 4.0",
        "n_sequences": len(collected),
        "caveat": (
            "absence of the antimicrobial keyword is not evidence of absence of "
            "antimicrobial activity; this is a BACKGROUND set, not a validated negative set"
        ),
        "sequences": collected,
    }, indent=2))
    print(f"  uniprot background: {len(collected)} sequences -> {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    ap.add_argument("--with-background", action="store_true",
                    help="also fetch the UniProt background set")
    ap.add_argument("--all-dramp", action="store_true",
                    help="fetch every listed DRAMP file, not just general_amps")
    args = ap.parse_args()

    print("Q-Peptide data download")
    print("DRAMP is CC BY 4.0; see research/papers.md section 6 for citations.\n")
    try:
        print("DRAMP:")
        download_dramp(list(DRAMP_FILES) if args.all_dramp else None, force=args.force)
        if args.with_background:
            print("UniProt:")
            download_uniprot_background(force=args.force)
    except requests.RequestException as exc:
        print(f"\nDownload failed: {exc}", file=sys.stderr)
        print("Check your network connection, or download manually from "
              "http://dramp.cpu-bioinfor.org/downloads/ into data/raw/", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print("\nNext: python -m backend.data.curate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
