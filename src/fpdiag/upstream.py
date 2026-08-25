from __future__ import annotations

from pathlib import Path
import subprocess


def clone_upstream(url: str, destination: str | Path) -> str:
    destination = Path(destination)
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True)
    return subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def download_official_publish_log(repo_id: str, filename: str, cache_dir: str | Path):
    """Download only the small official publish log, never the dataset snapshot."""
    from huggingface_hub import dataset_info, hf_hub_download

    revision = dataset_info(repo_id).sha
    path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename,
                           revision=revision, cache_dir=str(cache_dir))
    return Path(path), revision
