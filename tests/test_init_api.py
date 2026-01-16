# tests/test_init_api.py
from __future__ import annotations

import importlib
import pytest
import sys
sys.path.append("./src")


def test_import_package_root_is_stable():
    """
    Importing the package root should never fail just because optional/ML modules
    are missing or broken. This is the main guarantee of the layered __init__.py.
    """
    qsb = importlib.import_module("Q_Sea_Battle")
    assert hasattr(qsb, "__all__")
    assert isinstance(qsb.__all__, list)
    assert "GameLayout" in qsb.__all__


def test_core_exports_are_present_and_accessible():
    """
    Core API must exist immediately after importing Q_Sea_Battle.
    """
    import Q_Sea_Battle as qsb

    core = [
        "GameLayout",
        "GameEnv",
        "Players",
        "PlayerA",
        "PlayerB",
        "Game",
        "Tournament",
        "TournamentLog",
        "SimplePlayers",
        "MajorityPlayers",
        "PRAssisted",
        "PRAssistedLayer",
        "PRAssistedPlayers",
        "PRAssistedPlayerA",
        "PRAssistedPlayerB",
        "binary_entropy",
        "binary_entropy_reverse",
        "expected_win_rate_simple",
        "expected_win_rate_majority",
        "expected_win_rate_assisted",
        "limit_from_mutual_information",
        "logit_to_prob",
        "logit_to_logprob",
        "dru_train",
        "dru_execute",
    ]

    missing = [name for name in core if not hasattr(qsb, name)]
    assert missing == []


def test_all_names_resolve_or_raise_clean_import_error():
    """
    For every name in __all__:
      - either it is accessible (core or optional available),
      - or it fails with a clean ImportError/ModuleNotFoundError due to missing optional deps.

    This ensures __all__ doesn't contain typos/ghost exports.
    """
    import Q_Sea_Battle as qsb

    bad = []
    optional_missing = []

    for name in qsb.__all__:
        try:
            getattr(qsb, name)
        except (ImportError, ModuleNotFoundError) as e:
            # Optional dependency path: acceptable.
            optional_missing.append((name, type(e).__name__, str(e)))
        except AttributeError as e:
            # This is a real bug: __all__ lists a name that cannot be resolved.
            bad.append((name, type(e).__name__, str(e)))

    assert bad == [], f"__all__ contains unresolvable names: {bad}"


@pytest.mark.parametrize(
    "lazy_name",
    [
        # These should exist as attributes, but may raise ImportError if TF isn't present.
        "NeuralNetPlayers",
        "NeuralNetPlayerA",
        "NeuralNetPlayerB",
        "TrainableAssistedPlayers",
        "TrainableAssistedPlayerA",
        "TrainableAssistedPlayerB",
        "LinTrainableAssistedModelA",
        "LinTrainableAssistedModelB",
        "PyrTrainableAssistedModelA",
        "PyrTrainableAssistedModelB",
    ],
)
def test_lazy_exports_are_declared(lazy_name: str):
    """
    The layered __init__.py should declare these names in __all__ even if they are lazy.
    """
    import Q_Sea_Battle as qsb

    assert lazy_name in qsb.__all__


def test_lin_pyr_imitation_exports_are_namespaced():
    """
    Ensure no ambiguity between Lin and Pyr imitation utility functions.
    """
    import Q_Sea_Battle as qsb

    assert "lin_generate_measurement_dataset_a" in qsb.__all__
    assert "pyr_generate_measurement_dataset_a" in qsb.__all__
    assert "generate_measurement_dataset_a" not in qsb.__all__
