"""
Downloads NIH ChestX-ray14 PNG images from Kaggle.

Uses Kaggle SDK directly with thread-local API instances for concurrent downloads.
Run: python -m src.data.download --config experiments/configs/base_config.yaml
"""
import argparse
import json
import sys
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

OWNER = "nih-chest-xrays"
DATASET = "data"

# Thread-local storage so each worker thread gets its own KaggleApi instance
_thread_local = threading.local()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_credentials() -> tuple:
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    creds = json.loads(cred_path.read_text())
    return creds["username"], creds["key"]


def get_api():
    """Return a thread-local authenticated KaggleApi instance."""
    if not hasattr(_thread_local, "api"):
        from kaggle.api.kaggle_api_extended import KaggleApi
        _thread_local.api = KaggleApi()
        _thread_local.api.authenticate()
    return _thread_local.api


def run_kaggle_files(slug: str, page_token: str = None) -> tuple:
    cmd = [sys.executable, "-m", "kaggle", "datasets", "files", slug]
    if page_token:
        cmd += ["--page-token", page_token]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return [], None

    lines = result.stdout.strip().splitlines()
    paths, next_token = [], None

    for line in lines:
        line = line.strip()
        if line.startswith("Next Page Token"):
            next_token = line.split("=", 1)[-1].strip()
            continue
        if not line or line.startswith("name") or line.startswith("---"):
            continue
        name = line.split()[0]
        paths.append(name)

    return paths, next_token


def list_target_files(slug: str, target_folders: list) -> list:
    """Single pass — stops once all target folders are done."""
    collected = []
    page_token = None
    page = 0
    seen_folders = set()
    target_set = set(target_folders)

    print(f"  Listing files (stops after folders: {target_folders}) ...")
    while True:
        paths, next_token = run_kaggle_files(slug, page_token)
        for name in paths:
            top = name.split("/")[0] if "/" in name else ""
            if top in target_set:
                collected.append(name)
                seen_folders.add(top)
            elif top and top > max(target_set):
                # Alphabetically past all targets — stop
                print(f"  Passed all target folders at page {page}.")
                print(f"  Total files found: {len(collected):,}")
                return collected

        page += 1
        if page % 50 == 0:
            print(f"    Page {page}: {len(collected):,} collected "
                  f"(seen: {sorted(seen_folders)})")

        if not next_token:
            break
        page_token = next_token

    print(f"  Total files found: {len(collected):,}")
    return collected


def download_one_sdk(file_path: str, dest_dir: Path) -> bool:
    """Download a single file using the thread-local KaggleApi instance."""
    out_path = dest_dir / Path(file_path).name
    if out_path.exists():
        return True
    try:
        api = get_api()
        api.dataset_download_file(
            f"{OWNER}/{DATASET}", file_path,
            path=str(dest_dir), quiet=True, force=False,
        )
        return out_path.exists()
    except Exception:
        return False


def download_all(file_paths: list, dest_dir: Path, workers: int = 8) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    success = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one_sdk, fp, dest_dir): fp
            for fp in file_paths
        }
        with tqdm(total=len(futures), unit="img", desc="  Downloading") as bar:
            for fut in as_completed(futures):
                if fut.result():
                    success += 1
                bar.update(1)

    return success


def download_csv_file(slug: str, filename: str, dest: Path) -> None:
    if dest.exists():
        print(f"  Already downloaded: {filename}")
        return
    print(f"  Downloading {filename} ...")
    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", slug, "-f", filename, "-p", str(dest.parent)],
        capture_output=True,
    )
    if dest.exists():
        print(f"  Saved: {filename}")
    else:
        print(f"  Warning: could not download {filename}")


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    username, key = load_credentials()
    owner, dataset_name = cfg["data"]["kaggle_dataset"].split("/")
    slug = f"{owner}/{dataset_name}"
    print(f"Authenticated as: {username}\n")

    # ── Metadata & splits ─────────────────────────────────────────────────────
    print("[1/3] Metadata & split files")
    download_csv_file(slug, cfg["data"]["metadata_file"],
                      raw_dir / cfg["data"]["metadata_file"])
    download_csv_file(slug, "train_val_list.txt", raw_dir / "train_val_list.txt")
    download_csv_file(slug, "test_list.txt", raw_dir / "test_list.txt")

    # ── List target image files ────────────────────────────────────────────────
    print("\n[2/3] Listing image files ...")
    target_folders = [f.replace(".zip", "") for f in cfg["data"]["archives_to_download"]]
    all_paths = list_target_files(slug, target_folders)

    if not all_paths:
        print("No files found.")
        return

    # ── Download ───────────────────────────────────────────────────────────────
    already = len(list(images_dir.glob("*.png")))
    remaining = [p for p in all_paths
                 if not (images_dir / Path(p).name).exists()]
    print(f"\n[3/3] {already:,} already downloaded, {len(remaining):,} remaining")

    if remaining:
        count = download_all(remaining, images_dir, workers=8)
        print(f"  Downloaded {count:,} new images")

    total = len(list(images_dir.glob("*.png")))
    print(f"\nDone. {total:,} total PNGs in {images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/base_config.yaml")
    args = parser.parse_args()
    main(args.config)
