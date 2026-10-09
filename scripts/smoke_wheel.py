"""Run outside the source checkout after installing the built wheel."""

import json
import subprocess
import sys
from pathlib import Path

import press
from press.config import Settings
from press.dataset.loader import load_dataset, validate_dataset

checkout = Path(__file__).resolve().parents[1]
package = Path(press.__file__).resolve()
assert not package.is_relative_to(checkout / "press"), f"Imported source checkout: {package}"
settings = Settings(_env_file=None)
manifest = load_dataset(settings.dataset_path)
assert manifest.total_questions == 500
assert validate_dataset(manifest) == []
plan = subprocess.run(
    [sys.executable, "-m", "press.cli", "run", "--model", "gpt-test", "--limit", "2", "--dry-run"],
    check=True,
    text=True,
    capture_output=True,
)
assert json.loads(plan.stdout)["questions"] == 2
subprocess.run([sys.executable, "-m", "press.cli", "dataset", "validate"], check=True)
print(f"Installed-wheel smoke passed: {package}")
