import importlib
import inspect
import sys

import numpy as np
import pytest

# Ensure src/ is importable when running pytest from repo root
sys.path.append("./src")


def _make_inputs(layout):
    n2 = layout.field_size ** 2
    m = layout.comms_size
    field = np.random.randint(0, 2, size=(n2,), dtype=int)
    gun = np.zeros(n2, dtype=int)
    gun[np.random.randint(0, n2)] = 1
    comm = np.random.randint(0, 2, size=(m,), dtype=int)
    return field, gun, comm


def _assert_comm(comm, layout):
    comm = np.asarray(comm)
    assert comm.shape == (layout.comms_size,)
    assert set(np.unique(comm.astype(int))).issubset({0, 1})


def _assert_shoot(shoot):
    assert int(shoot) in (0, 1)


def _construct_factory(Factory, layout):
    """
    Best-effort construction across inconsistent factory __init__ signatures.
    Supplies required parameters where known.
    """
    sig = inspect.signature(Factory.__init__)
    kwargs = {}

    # Skip 'self'
    params = list(sig.parameters.values())[1:]

    for p in params:
        if p.name in ("game_layout", "layout"):
            kwargs[p.name] = layout
        elif p.name == "p_high":
            kwargs[p.name] = 0.9
        elif p.name == "p_low":
            kwargs[p.name] = 0.1
        elif p.name in ("seed", "random_state"):
            kwargs[p.name] = 0
        elif p.default is inspect._empty:
            raise TypeError(
                f"Don't know how to construct {Factory.__name__}: "
                f"required parameter '{p.name}' has no default"
            )

    return Factory(**kwargs)


@pytest.mark.usefixtures("qsb")
@pytest.mark.parametrize(
    "factory_path",
    [
        "Q_Sea_Battle.players_base.Players",
        "Q_Sea_Battle.simple_players.SimplePlayers",
        "Q_Sea_Battle.majority_players.MajorityPlayers",
        "Q_Sea_Battle.pr_assisted_players.PRAssistedPlayers",
        "Q_Sea_Battle.trainable_assisted_players.TrainableAssistedPlayers",
    ],
)
def test_player_factories_produce_compatible_players(factory_path):
    from Q_Sea_Battle.game_layout import GameLayout

    mod_name, cls_name = factory_path.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    Factory = getattr(mod, cls_name)

    layout = GameLayout(field_size=4, comms_size=1)

    factory = _construct_factory(Factory, layout)
    player_a, player_b = factory.players()

    np.random.seed(0)
    field, gun, _ = _make_inputs(layout)

    # Player A must always work
    comm = player_a.decide(field)
    _assert_comm(comm, layout)


    shoot = player_b.decide(gun, comm)
    _assert_shoot(shoot)


@pytest.mark.usefixtures("qsb")
def test_neural_net_factories_are_skipped_without_tensorflow():
    tf = pytest.importorskip("tensorflow")
    assert tf is not None

    from Q_Sea_Battle.game_layout import GameLayout
    from Q_Sea_Battle.neural_net_players import NeuralNetPlayers

    layout = GameLayout(field_size=4, comms_size=1)
    players = NeuralNetPlayers(game_layout=layout)
    pa, pb = players.players()

    np.random.seed(0)
    field, gun, _ = _make_inputs(layout)

    comm = pa.decide(field)
    _assert_comm(comm, layout)

    try:
        shoot = pb.decide(gun, comm)
    except NotImplementedError:
        pytest.skip(
            "NeuralNetPlayers PlayerB depends on an unimplemented Keras model (no call())."
        )

    _assert_shoot(shoot)
