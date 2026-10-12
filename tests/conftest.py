import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

if not (REPO_ROOT / "data" / "config" / "nodes").exists():
    pytest.skip("run scripts/build_consumption_profiles.py first", allow_module_level=True)
