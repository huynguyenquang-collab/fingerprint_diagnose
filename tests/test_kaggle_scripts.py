from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_scripts_are_strict_and_bootstrap_does_not_install_torch():
    scripts = list((ROOT / "scripts").glob("*.sh"))
    assert len(scripts) == 5
    assert all("set -euo pipefail" in script.read_text() for script in scripts)
    assert "pip install torch" not in (ROOT / "scripts/kaggle_bootstrap.sh").read_text()


def test_readme_has_six_cells_and_archive_path():
    readme = (ROOT / "README.md").read_text()
    assert all(f"Cell {i}" in readme for i in range(1, 7))
    assert "/kaggle/working/if_sft_fpdiag_results.zip" in readme
