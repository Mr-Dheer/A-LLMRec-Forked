"""
Download product images for A-LLMRec datasets.

Derives the valid ASIN set by replaying the same two-pass filtering logic
used in pre_train/sasrec/data_preprocess.py, then downloads one image per
ASIN from the Amazon metadata JSON.

Usage:
    python data/images/download_images.py \\
      --dataset All_Beauty \\
      --reviews data/amazon/All_Beauty.json.gz \\
      --metadata data/amazon/meta_All_Beauty.json \\
      --output_dir data/images/All_Beauty \\
      --workers 8
"""

import argparse
import csv
import gzip
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Step 1: Derive valid ASINs by replaying data_preprocess.py filter logic
# ---------------------------------------------------------------------------

def derive_valid_asins(reviews_path: Path, dataset: str) -> set[str]:
    """
    Two-pass over the gzipped review file, mirroring data_preprocess.py:
      - Skip reviews with overall < 3 for Beauty/Toys datasets
      - Threshold = 4 for Beauty/Toys (5 otherwise)
      - An ASIN is valid iff countP[asin] >= threshold AND at least one
        qualifying reviewer also has countU[reviewer] >= threshold
    """
    is_beauty_or_toys = ("Beauty" in dataset) or ("Toys" in dataset)
    threshold = 4 if is_beauty_or_toys else 5

    print(f"\n[Step 1] Scanning {reviews_path} to derive valid ASINs ...")
    print("  First 5 review records (for format verification):")

    countU: dict[str, int] = defaultdict(int)
    countP: dict[str, int] = defaultdict(int)

    # Pass 1: count interactions
    shown = 0
    with gzip.open(reviews_path, "rb") as f:
        for raw in tqdm(f, desc="  Pass 1 (counting)", unit="line"):
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if shown < 5:
                print(f"    {json.dumps({k: rec.get(k) for k in ('asin','reviewerID','overall','unixReviewTime')})}")
                shown += 1
            if is_beauty_or_toys and rec.get("overall", 5) < 3:
                continue
            countU[rec["reviewerID"]] += 1
            countP[rec["asin"]] += 1

    # Pass 2: collect ASINs that appear in at least one qualifying interaction
    valid_asins: set[str] = set()
    with gzip.open(reviews_path, "rb") as f:
        for raw in tqdm(f, desc="  Pass 2 (filtering)", unit="line"):
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if is_beauty_or_toys and rec.get("overall", 5) < 3:
                continue
            asin = rec["asin"]
            reviewer = rec["reviewerID"]
            if countU[reviewer] >= threshold and countP[asin] >= threshold:
                valid_asins.add(asin)

    print(f"\n  Derived {len(valid_asins)} valid ASINs from review data.")
    return valid_asins


# ---------------------------------------------------------------------------
# Step 2: Match ASINs to image URLs from metadata (streaming, line-by-line)
# ---------------------------------------------------------------------------

def load_image_urls(metadata_path: Path, valid_asins: set[str]) -> dict[str, str]:
    """
    Stream meta JSON line-by-line. For each matching ASIN prefer
    imageURLHighRes[0], fall back to imageURL[0].
    """
    print(f"\n[Step 2] Scanning {metadata_path} for image URLs ...")
    url_map: dict[str, str] = {}
    total_lines = 0

    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="  Scanning metadata", unit="line"):
            line = line.strip()
            if not line:
                continue
            total_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            asin = obj.get("asin", "")
            if asin not in valid_asins:
                continue

            hi = obj.get("imageURLHighRes") or []
            lo = obj.get("imageURL") or []

            url = ""
            if hi and hi[0]:
                url = hi[0]
            elif lo and lo[0]:
                url = lo[0]

            url_map[asin] = url  # empty string if no URL found

    matched = sum(1 for v in url_map.values() if v)
    print(f"  Matched {matched} / {len(valid_asins)} ASINs to image URLs "
          f"(metadata had {total_lines} products total).")
    return url_map


# ---------------------------------------------------------------------------
# Step 3: Download a single image
# ---------------------------------------------------------------------------

def download_one(
    asin: str,
    url: str,
    output_dir: Path,
    timeout: int = 10,
) -> tuple[str, str, str, str]:
    """
    Returns (asin, status, url_used, file_path).
    status: 'success' | 'failed_download' | 'failed_validation' | 'no_url_in_metadata'
    """
    dest = output_dir / f"{asin}.jpg"

    if not url:
        return asin, "no_url_in_metadata", "", ""

    # Resume: skip already-validated files
    if dest.exists():
        try:
            with Image.open(dest) as img:
                img.verify()
            return asin, "success", url, str(dest)
        except Exception:
            dest.unlink(missing_ok=True)

    # Random courtesy delay
    time.sleep(random.uniform(0.1, 0.3))

    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or len(resp.content) <= 500:
            return asin, "failed_download", url, ""

        dest.write_bytes(resp.content)
    except Exception as exc:
        return asin, "failed_download", url, ""

    # PIL validation
    try:
        with Image.open(dest) as img:
            img.verify()
        return asin, "success", url, str(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        return asin, "failed_validation", url, ""


# ---------------------------------------------------------------------------
# Step 4: Orchestrate parallel downloads and write CSV report
# ---------------------------------------------------------------------------

def run_downloads(
    valid_asins: set[str],
    url_map: dict[str, str],
    output_dir: Path,
    report_dir: Path,
    dataset: str,
    workers: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / f"{dataset}_report.csv"
    asins_list = sorted(valid_asins)

    print(f"\n[Step 3] Downloading {len(asins_list)} images with {workers} workers ...")

    rows: list[tuple[str, str, str, str]] = []
    success = failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_one, asin, url_map.get(asin, ""), output_dir): asin
            for asin in asins_list
        }
        with tqdm(total=len(asins_list), unit="img", desc="  Downloading") as bar:
            for fut in as_completed(futures):
                asin, status, url_used, file_path = fut.result()
                rows.append((asin, status, url_used, file_path))
                if status == "success":
                    success += 1
                else:
                    failed += 1
                    if status != "no_url_in_metadata":
                        tqdm.write(f"  [WARN] {status}: {asin}  {url_used}")
                bar.update(1)

    # Write CSV (sorted by asin for reproducibility)
    rows.sort(key=lambda r: r[0])
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["asin", "status", "url_used", "file_path"])
        writer.writerows(rows)

    total = len(asins_list)
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}")
    print(f"Total items:        {total}")
    print(f"Successfully saved: {success:>5}  ({100*success/total:.1f}%)")
    print(f"Failed/missing:     {failed:>5}  ({100*failed/total:.1f}%)")
    print(f"Report written to:  {report_path}")
    print(f"{'='*50}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download product images for A-LLMRec datasets."
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name, e.g. All_Beauty")
    parser.add_argument("--reviews", required=True,
                        help="Path to the gzipped review file, e.g. data/amazon/All_Beauty.json.gz")
    parser.add_argument("--metadata", required=True,
                        help="Path to the metadata JSON (one object per line), e.g. data/amazon/meta_All_Beauty.json")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save downloaded images")
    parser.add_argument("--report_dir", default=None,
                        help="Directory for the CSV coverage report (default: <output_dir>/../download_report)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download threads (default: 8)")
    parser.add_argument("--timeout", type=int, default=10,
                        help="HTTP request timeout in seconds (default: 10)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir) if args.report_dir else output_dir.parent / "download_report"

    valid_asins = derive_valid_asins(Path(args.reviews), args.dataset)
    url_map = load_image_urls(Path(args.metadata), valid_asins)
    run_downloads(valid_asins, url_map, output_dir, report_dir, args.dataset, args.workers)


if __name__ == "__main__":
    main()
