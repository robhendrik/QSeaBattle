import importlib
import pytest
import sys
sys.path.append("./src")
@pytest.mark.parametrize(
    "module_name",
    [
        "Q_Sea_Battle.game",
        "Q_Sea_Battle.game_env",
        "Q_Sea_Battle.game_layout",
        "Q_Sea_Battle.tournament",
        "Q_Sea_Battle.tournament_log",
        "Q_Sea_Battle.players_base",
        "Q_Sea_Battle.simple_players",
        "Q_Sea_Battle.majority_players",
        "Q_Sea_Battle.pr_assisted",
        "Q_Sea_Battle.pr_assisted_players",
        "Q_Sea_Battle.reference_performance_utilities",
        "Q_Sea_Battle.dru_utilities",
        "Q_Sea_Battle.logit_utilities",
        "Q_Sea_Battle.neural_net_imitation_utilities",
        "Q_Sea_Battle.pyr_dataset_generation_utilities",
        "Q_Sea_Battle.pyr_dataset_conversion_utilities",
    ],
)
def test_modules_import(module_name):
    importlib.import_module(module_name)

def test_tensorflow_dependent_modules_import_if_tensorflow_present():
    tf = pytest.importorskip("tensorflow")
    assert tf is not None

    for module_name in [
        "Q_Sea_Battle.neural_net_players",
        "Q_Sea_Battle.neural_net_player_a",
        "Q_Sea_Battle.neural_net_player_b",
        "Q_Sea_Battle.pr_assisted_replay",
        "Q_Sea_Battle.pyr_measurement_layer_a",
        "Q_Sea_Battle.pyr_measurement_layer_b",
        "Q_Sea_Battle.pyr_combine_layer_a",
        "Q_Sea_Battle.pyr_combine_layer_b",
        "Q_Sea_Battle.pyr_internal_model_a",
        "Q_Sea_Battle.pyr_internal_model_b",
    ]:
        importlib.import_module(module_name)
