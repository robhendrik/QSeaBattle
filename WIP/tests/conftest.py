import importlib
import sys
from pathlib import Path
import pytest

# Ensure WIP/src is on sys.path so tests in WIP/tests can import the module
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
WIP_SRC = REPO_ROOT / "WIP" / "src"
if str(WIP_SRC) not in sys.path:
    sys.path.insert(0, str(WIP_SRC))

@pytest.fixture(scope="session")
def pr_module():
    return importlib.import_module("Q_Sea_Battle_New.pr_assisted_replay")

@pytest.fixture
def PRAssistedReplay(pr_module):
    if not hasattr(pr_module, "PRAssistedReplay"):
        raise AttributeError("Q_Sea_Battle_New.pr_assisted_replay does not define PRAssistedReplay")
    return pr_module.PRAssistedReplay
