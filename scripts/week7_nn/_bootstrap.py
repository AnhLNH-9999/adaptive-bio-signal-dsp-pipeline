"""Make ``src`` importable when a script is run directly from anywhere in the repo."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
