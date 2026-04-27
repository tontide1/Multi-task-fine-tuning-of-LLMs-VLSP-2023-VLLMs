"""
Top-level conftest.py for pytest.

Ensures the repository root is on sys.path so that pytest can collect
`scripts/test_baseline_parsers.py` (and any future tests under `scripts/`)
and resolve `from scripts.<module>` imports regardless of the directory
pytest is invoked from.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
