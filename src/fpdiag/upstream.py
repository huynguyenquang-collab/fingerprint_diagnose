from pathlib import Path
import subprocess


def clone_upstream(url: str, destination: str | Path) -> str:
    destination = Path(destination)
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", url, str(destination)], check=True)
    return subprocess.run(["git", "-C", str(destination), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
