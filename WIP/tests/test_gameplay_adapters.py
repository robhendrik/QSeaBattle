# """
# Tests for gameplay_adapters.py.

# We test:
# - Adapter A returns comm_logits unchanged and meas/out lists as binary {0,1}.
# - Adapter B accepts gameplay binary inputs and lists and returns shoot_logit.
# - Adapters work with a Pyr internal model (depth>1) AND a minimal Lin-like internal model (depth=1).

# Repo layout:
# QSeaBattle/
#   WIP/src/Q_Sea_Battle_New/...
#   WIP/tests/test_gameplay_adapters.py
# """

# import sys
# from pathlib import Path
# import tensorflow as tf

# ROOT = Path(__file__).resolve().parents[2]
# WIP_SRC = ROOT / "WIP" / "src"
# sys.path.insert(0, str(WIP_SRC))

# from Q_Sea_Battle_New.gameplay_adapters import GameplayModelAAdapter, GameplayModelBAdapter
# from Q_Sea_Battle_New.pyr_internal_model_a import PyrInternalModelA
# from Q_Sea_Battle_New.pyr_internal_model_b import PyrInternalModelB
# from Q_Sea_Battle_New.pyr_dataset_generation_utilities import generate_pyr_dataset


# class _Layout:
#     def __init__(self, n2: int):
#         self.n2 = n2
#         self.comms_size = 1


# def _is_binary(x: tf.Tensor) -> bool:
#     x = tf.convert_to_tensor(x)
#     return bool(tf.reduce_all(tf.logical_or(tf.equal(x, 0.0), tf.equal(x, 1.0))).numpy())


# def test_pyr_adapters_end_to_end_shapes_and_domains():
#     layout = _Layout(16)

#     # IMPORTANT: For gameplay, internal A must not be sr_mode="replay" unless you also provide replay outcomes.
#     model_a = PyrInternalModelA(layout, sr_mode="stochastic", beta=10.0, alpha=5.0, seed=0)
#     model_b = PyrInternalModelB(layout, sr_mode="replay", beta=10.0, alpha=5.0, seed=1)

#     a = GameplayModelAAdapter(model_a, beta_field=10.0)
#     b = GameplayModelBAdapter(model_b, beta_gun=10.0, beta_comm=10.0)

#     ds = generate_pyr_dataset(n2=16, num_games=1, seed=123, validate=True)
#     field_bits = tf.convert_to_tensor(ds["field_bits"][0:1, 0, :], dtype=tf.float32)
#     gun_bits = tf.convert_to_tensor(ds["gun_bits"][0:1, 0, :], dtype=tf.float32)

#     comm_logits, meas_list, out_list = a.compute_with_internal(field_bits)

#     assert comm_logits.shape == (1, 1)
#     assert comm_logits.dtype == tf.float32
#     assert isinstance(meas_list, list) and isinstance(out_list, list)
#     assert len(meas_list) == model_a.depth
#     assert len(out_list) == model_a.depth

#     for d in range(model_a.depth):
#         assert meas_list[d].shape.rank == 2
#         assert out_list[d].shape.rank == 2
#         assert meas_list[d].dtype == tf.float32
#         assert out_list[d].dtype == tf.float32
#         #assert _is_binary(meas_list[d])
#         #assert _is_binary(out_list[d])

#     comm_bits = tf.cast(comm_logits >= 0.0, tf.float32)
#     shoot_logit = b([gun_bits, comm_bits, meas_list, out_list])

#     assert shoot_logit.shape == (1, 1)
#     assert shoot_logit.dtype == tf.float32


# def test_lin_like_internal_models_reuse_same_adapters():
#     class LinInternalA:
#         def __init__(self, n2: int):
#             self.n2 = n2
#             self.M = 1
#             self.depth = 1

#         def compute_with_internal(self, field_scaled, replay_out_a_logits_list=None, training=False):
#             B = tf.shape(field_scaled)[0]
#             comm_logits = tf.reduce_sum(field_scaled, axis=-1, keepdims=True)
#             meas_logits = tf.zeros((B, 1), tf.float32)
#             out_logits = tf.zeros((B, 1), tf.float32)
#             return comm_logits, [meas_logits], [out_logits]

#     class LinInternalB:
#         def __init__(self, n2: int):
#             self.n2 = n2
#             self.M = 1
#             self.depth = 1

#         def __call__(self, inputs, training=False):
#             gun_scaled, comm_in, prev_meas_list, prev_out_list = inputs
#             return tf.identity(comm_in)

#     model_a = LinInternalA(n2=4)
#     model_b = LinInternalB(n2=4)

#     a = GameplayModelAAdapter(model_a)
#     b = GameplayModelBAdapter(model_b)

#     field_bits = tf.constant([[0.0, 1.0, 0.0, 1.0]], tf.float32)
#     gun_bits = tf.constant([[0.0, 0.0, 1.0, 0.0]], tf.float32)

#     comm_logits, meas_list, out_list = a.compute_with_internal(field_bits)

#     assert comm_logits.shape == (1, 1)
#     assert len(meas_list) == 1 and len(out_list) == 1
#     assert _is_binary(meas_list[0]) and _is_binary(out_list[0])

#     comm_bits = tf.cast(comm_logits >= 0.0, tf.float32)
#     shoot_logit = b([gun_bits, comm_bits, meas_list, out_list])

#     assert shoot_logit.shape == (1, 1)
#     assert shoot_logit.dtype == tf.float32


# def test_players_tournament_integration_smoke():
#     """
#     Smoke test:
#     - Instantiate Pyr internal models (A,B) with GameLayout from core package.
#     - Wrap them with generic gameplay adapters.
#     - Plug adapters into TrainableAssistedPlayers (core Players wrapper).
#     - Run a small Tournament.
#     """
#     # Add core package src to path (QSeaBattle/src)
#     CORE_SRC = ROOT / "src"
#     sys.path.insert(0, str(CORE_SRC))

#     from Q_Sea_Battle.game_layout import GameLayout
#     from Q_Sea_Battle.game_env import GameEnv
#     from Q_Sea_Battle.trainable_assisted_players import TrainableAssistedPlayers
#     from Q_Sea_Battle.tournament import Tournament

#     layout = GameLayout(
#         field_size=4,          # n2=16 (power of two)
#         comms_size=1,          # pyr requires m=1
#         enemy_probability=0.5,
#         channel_noise=0.0,
#         number_of_games_in_tournament=5,
#     )

#     # Internal models live in WIP/src/Q_Sea_Battle_New
#     internal_a = PyrInternalModelA(layout, sr_mode="stochastic", beta=10.0, alpha=5.0, seed=0)
#     internal_b = PyrInternalModelB(layout, sr_mode="replay", beta=10.0, alpha=5.0, seed=1)

#     model_a = GameplayModelAAdapter(internal_a, beta_field=10.0)
#     model_b = GameplayModelBAdapter(internal_b, beta_gun=10.0, beta_comm=10.0)

#     # Wire into core Players wrapper (this will create TrainableAssistedPlayerA/B wrappers)
#     players = TrainableAssistedPlayers(layout, model_a=model_a, model_b=model_b)
#     env = GameEnv(layout)

#     t = Tournament(game_env=env, players=players, game_layout=layout)
#     log = t.tournament()

#     assert log is not None
