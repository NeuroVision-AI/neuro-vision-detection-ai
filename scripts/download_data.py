"""
Script to download and organize the Kaggle Brain Tumor MRI Dataset.
Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Requirements:
- `pip install kaggle` (public datasets can be downloaded anonymously)
"""

import os
import sys
from pathlib import Path
import subprocess
import argparse
from datetime import datetime, timezone

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config

KAGGLE_DATASET = "masoudnickparvar/brain-tumor-mri-dataset"


def kaggle_runtime() -> tuple[str, dict]:
    """Return a project-local Kaggle executable and public-download environment."""
    executable = Path(sys.executable).parent / "kaggle"
    command = str(executable) if executable.exists() else "kaggle"
    env = os.environ.copy()
    config_dir = config.DATA_DIR / ".kaggle"
    config_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("KAGGLE_CONFIG_DIR", str(config_dir))
    # Kaggle's public endpoints permit anonymous access, while the CLI still
    # expects these fields to exist when constructing its client.
    env.setdefault("KAGGLE_USERNAME", "anonymous")
    env.setdefault("KAGGLE_KEY", "anonymous")
    return command, env


def download_kaggle_dataset(raw_dir: Path) -> bool:
    """Download and unzip the dataset using Kaggle CLI."""
    print(f"\n📥 Downloading {KAGGLE_DATASET} from Kaggle...")
    command, env = kaggle_runtime()

    try:
        # Check if kaggle command exists
        subprocess.run([command, "--version"], check=True, capture_output=True, env=env)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: 'kaggle' CLI tool not found. Install it with: pip install kaggle")
        return False

    try:
        # Download
        subprocess.run([
            command, "datasets", "download", "-d", KAGGLE_DATASET,
            "-p", str(raw_dir), "--unzip"
        ], check=True, env=env)
        print(f"✅ Successfully downloaded dataset to {raw_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        return False


def organize_kaggle_data(
    raw_dir: Path,
    source_name: str,
    exclude_conflicting_groups: bool,
) -> None:
    """
    The Kaggle dataset comes with pre-split 'Training' and 'Testing' folders.
    Preserve its official test split and group duplicates before creating a
    validation split. This avoids contaminating evaluation by merging and
    randomly re-splitting the official test images.
    """
    print("\n📂 Organizing downloaded dataset...")
    from src.data_integrity import prepare_research_dataset

    print("\n✂️ Creating provenance-aware grouped splits...")
    audit = prepare_research_dataset(
        source_root=raw_dir,
        output_root=config.PROCESSED_DATA_DIR,
        manifest_dir=config.MANIFEST_DIR,
        source_name=source_name,
        patient_pattern=None,
        seed=config.RANDOM_SEED,
        preserve_source_test=True,
        overwrite=False,
        exclude_conflicting_groups=exclude_conflicting_groups,
    )
    print(f"  Leakage-free: {audit['leakage_free']}")
    print(f"  Split counts: {audit['assigned_split_counts']}")
    print(f"\n✅ Data fully prepared and saved to {config.PROCESSED_DATA_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Download and organize Kaggle dataset")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download and only organize existing data")
    parser.add_argument(
        "--source-version-label",
        default=None,
        help="Pinned source/version label; defaults to a retrieval-date label",
    )
    parser.add_argument(
        "--exclude-conflicting-groups",
        action="store_true",
        help="Exclude and log entire duplicate groups that carry contradictory labels",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("  AI NeuroOnco — Data Downloader (Track 3)")
    print("=" * 50)

    if not args.skip_download:
        if not download_kaggle_dataset(config.RAW_DATA_DIR):
            return

    retrieval_date = datetime.now(timezone.utc).date().isoformat()
    source_name = args.source_version_label or (
        f"kaggle_masoudnickparvar_brain_tumor_mri_latest_retrieved_{retrieval_date}"
    )
    organize_kaggle_data(
        config.RAW_DATA_DIR,
        source_name,
        args.exclude_conflicting_groups,
    )

    print("\n🚀 Next step: Run 'python -m src.train --model MODEL_NAME' to train a model!")


if __name__ == "__main__":
    main()
