import importlib
import sys
from pathlib import Path
import pytest

# Ensure WIP/src is on sys.path so tests in WIP/tests can import the module
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
WIP_SRC = REPO_ROOT / "WIP" / "src"
SRC = REPO_ROOT / "src"
if str(WIP_SRC) not in sys.path:
    sys.path.insert(0, str(WIP_SRC))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

@pytest.fixture(scope="session")
def pr_module():
    return importlib.import_module("Q_Sea_Battle_New.pr_assisted_replay")

@pytest.fixture
def PRAssistedReplay(pr_module):
    if not hasattr(pr_module, "PRAssistedReplay"):
        raise AttributeError("Q_Sea_Battle_New.pr_assisted_replay does not define PRAssistedReplay")
    return pr_module.PRAssistedReplay


@pytest.fixture(scope="session")
def qsb():
    """Import helper for Q_Sea_Battle package in WIP test tree."""
    try:
        import Q_Sea_Battle  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(
            "Q_Sea_Battle package is not importable in this environment. "
            f"Import error: {e}"
        )
    import Q_Sea_Battle as pkg
    return pkg
