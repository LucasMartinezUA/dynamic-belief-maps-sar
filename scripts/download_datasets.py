#!/usr/bin/env python3
"""
SAREnv Heatmap and Features Downloader Pipeline

This script automates the retrieval of 'heatmap.npy' and 'features.geojson' files across 18 distinct
datasets from the official SAREnv GitHub repository. It ensures reproducible
research setups by pinning downloads to an immutable cryptographic commit hash.
It correctly routes LFS files (like GeoJSONs) through GitHub's media server.

Usage:
    pixi run python scripts/download_datasets.py
"""
import sys
from pathlib import Path
from urllib.request import Request, urlopen

# Directory Configuration
REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = REPO_ROOT / "sarenv_dataset"
BASE_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Repository Source Configuration
OWNER = "namurproject"
REPO = "SAREnv"
COMMIT = "0f2d344d17c0b9c18cec2176dd9b171beba727d6"
FILES_TO_DOWNLOAD = ["heatmap.npy", "features.geojson"]

# Target Datasets IDs
DATASET_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30]


def download_dataset_files(dataset_id: int):
    """Downloads the required files for a specific dataset ID from GitHub."""
    local_folder = BASE_DATA_DIR / str(dataset_id)
    local_folder.mkdir(exist_ok=True)

    for file_name in FILES_TO_DOWNLOAD:
        repo_path = f"sarenv_dataset/{dataset_id}/{file_name}"
        destination_path = local_folder / file_name

        if file_name.endswith(".geojson"):
            url_domain = "media.githubusercontent.com/media"
        else:
            url_domain = "raw.githubusercontent.com"

        raw_url = f"https://{url_domain}/{OWNER}/{REPO}/{COMMIT}/{repo_path}"


        # Skip download if the file already exists locally
        if destination_path.exists():
            print(f"Dataset {dataset_id:02d}: {file_name} already exists. Skipping.")
            continue

        print(f"Dataset {dataset_id:02d}: Downloading {file_name}...")
        try:
            req = Request(raw_url, headers={"User-Agent": "Dataset-Downloader"})
            
            with urlopen(req) as response:
                if response.status != 200:
                    raise Exception(f"HTTP Status {response.status}")
                
                with open(destination_path, "wb") as local_file:
                    local_file.write(response.read())
                    
            print(f"Dataset {dataset_id:02d}: {file_name} successfully saved.")
            
        except Exception as e:
            print(f"❌ Error downloading {file_name} for dataset {dataset_id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    print("Starting dataset download pipeline...")
    for d_id in DATASET_IDS:
        download_dataset_files(d_id)
    print("\nDataset download pipeline execution completed!")