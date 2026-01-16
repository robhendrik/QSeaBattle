import sys
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--qsb-import-path",
        action="store",
        default=None,
        help=(
            "Optional path to add to PYTHONPATH so the Q_Sea_Battle package can be imported "
            "(e.g., the repo root)."
        ),
    )


@pytest.fixture(scope="session", autouse=True)
def _maybe_add_qsb_path(request):
    path = request.config.getoption("--qsb-import-path")
    if path and path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def qsb():
    """Import helper.

    These tests are meant to live in the QSeaBattle repo. If the package isn't importable,
    we skip cleanly with a helpful message.
    """
    try:
        import Q_Sea_Battle  # noqa: F401
    except Exception as e:  # pragma: no cover
        pytest.skip(
            "Q_Sea_Battle package is not importable in this environment. "
            "Run pytest from the project repo root (or pass --qsb-import-path). "
            f"Import error: {e}"
        )
    import Q_Sea_Battle as pkg
    return pkg
