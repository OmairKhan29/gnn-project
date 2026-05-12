"""
Download SIDER and MUV datasets for Feature 2 transfer learning.
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from feature2.data.transfer_datasets import (
    download_transfer_dataset,
    load_transfer_dataset_csv,
    TRANSFER_DATASET_INFO,
)


def main():
    data_dir = "data/transfer"
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("Feature 2: Downloading Transfer Datasets")
    print("=" * 60)

    for dataset_name in ["sider", "muv"]:
        print(f"\n[{dataset_name.upper()}]")
        try:
            filepath = download_transfer_dataset(dataset_name, data_dir, verbose=True)
            df = load_transfer_dataset_csv(dataset_name, data_dir, verbose=True)
            info = TRANSFER_DATASET_INFO[dataset_name]
            print(f"  Tasks: {info['num_tasks']}")
            print(f"  Description: {info['description']}")
            print(f"  Status: ✅ Downloaded successfully")
        except Exception as e:
            print(f"  Status: ❌ Failed — {e}")
            print(f"  Manual fix: Download from the URL and place in {data_dir}/")

    print("\n" + "=" * 60)
    print("Download complete.")
    print(f"Data directory: {data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()