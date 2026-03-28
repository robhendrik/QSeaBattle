from __future__ import annotations

from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
TARGET_SCRIPT = THIS_DIR / "train_Phase_pre_gameplay_STE - CONFIGURABLE_multi_phase.py"
ROOT = THIS_DIR.parent

base_training_cfg = {
        # Geometry, this should not change in a training run.
        "N2": 16,
        "FIELD_SIZE": 4,
        "COMMS_SIZE": 1,
        "DEPTH": 4,
        "CHANNEL_NOISE": 0.0,
        "ENEMY_PROBABILITY": 0.5,

        # Training setup
        "SEED": 42,
        "BATCH": 32,
        "EPOCHS": 1000,
        "START_EPOCH": 0,

        # Directory structure
        "ROOT": ROOT,
        "WIP": ROOT / "WIP",
        "DATASET_DIR": ROOT / "WIP" /"dataset",
        "CHECKPOINT_DIR": ROOT / "WIP" / "checkpoints",
        "LOG_DIR": ROOT / "WIP" / "logs",
 
        # Training settings
        "LOAD_WEIGHTS_ON_START": False,
        "MODEL_A_WEIGHTS_IN": "checkpoints\\model_a_latest.weights.h5",
        "MODEL_B_WEIGHTS_IN": "checkpoints\\model_b_latest.weights.h5",
        "SAVE_WEIGHTS_EVERY": 100,
        "SAVE_WEIGHTS_AT_END": True,
        "LOG_FLUSH_EVERY_EPOCHS": 10,

        # Verbosity settings
        "VERBOSITY": 2, # 0 = no print, 1 = epoch summary, 2 = epoch summary + losses, 3 = detailed info including weight norms, gradient norms, etc.
        
        # LOG settings
        "LOG_NAME": None, # if None, will be auto-generated with timestamp
    }
base_interface_cfg = {
        # "mode": 2 for interp, "mode": 0 for none, "mode": 1 for hard.
        "comm": {"stop_gradient": False, 
                 "hardening": {"mode": 1, "beta": 10.0, "lambda": 0.0}, 
                 "noise_std": 0.30},
        "meas": {"stop_gradient": False, 
                 "hardening": {"mode": 1, "beta": 10.0, "lambda": 0.0}, 
                 "noise_std": 0.30},
        "out":  {"stop_gradient": False, 
                 "hardening": {"mode": 0, "beta": 10.0, "lambda": 0.0}, 
                 "noise_std": 0.30},
    }
base_early_stop_cfg = {
        "consecutive": 3,
        "rules": [
            {
                "shoot_acc": {"comparator": "gt", "value": 0.999}
            }
        ]
    }
base_stage_cfg = {
    "update_a" : False,
    "update_b" : True,
    "lr_a" : 1e-5,
    "lr_b" : 1e-5,
    
}
      
base_loss_cfg = {
    "w_meas_a_global": 0.2,
    "w_comms_b_global": 0.02,
    "w_meas_a_per_level" : [1.0, 1.0, 1.0, 1.0], 
    "w_comms_b_per_level" : [0.02, 0.02, 0.02], 
    "w_comm_a_global" : 0.05,
    "w_shoot" : 2.0,
    "w_comms_b_mag": float(0.0001),
    "w_comm_a_mag": float(0.0001),
    "w_meas_a_mag": float(0.0001),
    "w_meas_b_mag": float(0.0001),
    "mag_target" : 6.0,
}
base_rollback_cfg = {
        "ENABLE_ROLLBACK": False,
        "LOOKBACK": 5,
        "DROP_ABS": 0.02,
        "MIN_EPOCH": 50,
        "COOLDOWN_EPOCHS": 10,
    }
base_model_cfg = {
        "P_HIGH": 1.0,
        "MODEL_BETA": 10.0,
        "ALPHA_FOR_PR_LAYERS": 0.3,
    }

def _load_training_module(script_path: Path):
    spec = spec_from_file_location("qsb_multi_phase_training", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load training script from {script_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _phase(
    *,
    stage_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    interface_cfg: dict[str, Any],
    early_stop_cfg: dict[str, Any],
    rollback_cfg: dict[str, Any],
    max_epochs: int,
    load_weights_from: str | None = None,
    reset_optimizer: bool = True,
    model_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage_cfg": deepcopy(stage_cfg),
        "loss_cfg": deepcopy(loss_cfg),
        "interface_cfg": deepcopy(interface_cfg),
        "early_stop_cfg": deepcopy(early_stop_cfg),
        "rollback_cfg": deepcopy(rollback_cfg),
        "max_epochs": int(max_epochs),
        "load_weights_from": load_weights_from,
        "reset_optimizer": bool(reset_optimizer),
        "model_updates": deepcopy(model_updates or {}),
    }


def build_three_stage_plan() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Builds a conservative phase schedule that implements your requested curriculum:

    Stage 1:
      Train A only on comm_a + meas_a. B is frozen.

    Stage 2:
      Freeze A, train only B with A-provided inputs, curriculum on comm_b levels,
      then add shoot. Because train_multi_phase is flat, this stage is represented
      as four consecutive phases.

    Stage 3:
      Joint AB fine-tuning with mild meas_a target and main shoot target, while
      keeping magnitude targets on meas_a and meas_b with target 6.0.
    """

    training_cfg = deepcopy(base_training_cfg)
    rollback_cfg = deepcopy(base_rollback_cfg)
    interface_cfg = deepcopy(base_interface_cfg)
    model_cfg = deepcopy(base_model_cfg)
    early_stop_cfg = deepcopy(base_early_stop_cfg)

    # Optional: give this run its own log name so it does not blend with other runs.
    training_cfg["LOG_NAME"] = training_cfg.get("LOG_NAME") or "three_stage_curriculum"

    # Mildly safer interface defaults for staged training.
    # Keep comm interpolation, zero out runtime noise unless you explicitly want it.
    stage1_interface = deepcopy(base_interface_cfg)
    stage1_interface["comm"]["noise_std"] = 0.0
    stage1_interface["meas"]["noise_std"] = 0.0
    stage1_interface["out"]["noise_std"] = 0.0

    stage2_interface = deepcopy(stage1_interface)
    stage3_interface = deepcopy(stage1_interface)

    # ---------------------------
    # Stage 1: A only
    # ---------------------------
    stage1_stage_cfg = {
        "update_a": True,
        "update_b": False,
        "lr_a": 1e-3,
        "lr_b": 1e-5,
    }
    stage1_loss_cfg = {
        "w_meas_a_global": 1.0,
        "w_comms_b_global": 0.0,
        "w_meas_a_per_level": [1.0, 1.0, 1.0, 1.0],
        "w_comms_b_per_level": [0.0, 0.0, 0.0],
        "w_comm_a_global": 0.10,
        "w_shoot": 0.0,
        "w_comms_b_mag": float(0.0),
        "w_comm_a_mag": float(0.0001),
        "w_meas_a_mag": float(0.0003),
        "w_meas_b_mag": float(0.0),
        "mag_target": 6.0,
    }
    stage1_early_stop = {
        "consecutive": 3,
        "rules": [
            {
                "comm_a_loss": {"comparator": "lt", "value": 0.0002},
                "meas_in_a_loss": {"comparator": "lt", "value": 0.001},
            }
        ],
    }

    # ---------------------------
    # Stage 2: B only curriculum
    # ---------------------------
    stage2_stage_cfg = {
        "update_a": False,
        "update_b": True,
        "lr_a": 1e-5,
        "lr_b": 1e-4,
    }
    stage2_base_loss = {
        "w_meas_a_global": 0.0,
        "w_comms_b_global": 0.0,
        "w_meas_a_per_level": [0.0, 0.0, 0.0, 0.0],
        "w_comms_b_per_level": [0.0, 0.0, 0.0],
        "w_comm_a_global": 0.0,
        "w_shoot": 0.0,
        "w_comms_b_mag": float(0.0002),
        "w_comm_a_mag": float(0.0),
        "w_meas_a_mag": float(0.0),
        "w_meas_b_mag": float(0.0003),
        "mag_target": 6.0,
    }
    stage2_early_stop = {
        "consecutive": 3,
        "rules": [
            {
                "comms_b_loss": {"comparator": "lt", "value": 0.002}
            }
        ],
    }

    stage2_loss_lvl1 = deepcopy(stage2_base_loss)
    stage2_loss_lvl1["w_comms_b_global"] = 0.20
    stage2_loss_lvl1["w_comms_b_per_level"] = [1.0, 0.0, 0.0]

    stage2_loss_lvl12 = deepcopy(stage2_base_loss)
    stage2_loss_lvl12["w_comms_b_global"] = 0.25
    stage2_loss_lvl12["w_comms_b_per_level"] = [1.0, 1.0, 0.0]

    stage2_loss_lvl123 = deepcopy(stage2_base_loss)
    stage2_loss_lvl123["w_comms_b_global"] = 0.30
    stage2_loss_lvl123["w_comms_b_per_level"] = [1.0, 1.0, 1.0]

    stage2_loss_with_shoot = deepcopy(stage2_loss_lvl123)
    stage2_loss_with_shoot["w_shoot"] = 2.0

    # ---------------------------
    # Stage 3: joint AB
    # ---------------------------
    stage3_stage_cfg = {
        "update_a": True,
        "update_b": True,
        "lr_a": 5e-5,
        "lr_b": 5e-5,
    }
    stage3_loss_cfg = {
        "w_meas_a_global": 0.08,
        "w_comms_b_global": 0.10,
        "w_meas_a_per_level": [1.0, 1.0, 1.0, 1.0],
        "w_comms_b_per_level": [0.3, 0.3, 0.3],
        "w_comm_a_global": 0.05,
        "w_shoot": 2.5,
        "w_comms_b_mag": float(0.0002),
        "w_comm_a_mag": float(0.0001),
        "w_meas_a_mag": float(0.0003),
        "w_meas_b_mag": float(0.0003),
        "mag_target": 6.0,
    }
    stage3_early_stop = deepcopy(early_stop_cfg)

    # Apply runtime model updates through the model-level setters.
    default_model_updates = {
        "P_HIGH": model_cfg.get("P_HIGH", 1.0),
        "ALPHA_FOR_PR_LAYERS": model_cfg.get("ALPHA_FOR_PR_LAYERS", 0.3),
    }

    phases = [
        # Stage 1
        _phase(
            stage_cfg=stage1_stage_cfg,
            loss_cfg=stage1_loss_cfg,
            interface_cfg=stage1_interface,
            early_stop_cfg=stage1_early_stop,
            rollback_cfg=rollback_cfg,
            max_epochs=250,
            load_weights_from=None,
            reset_optimizer=True,
            model_updates=default_model_updates,
        ),
        # Stage 2.1 - B only, comm_b[1]
        _phase(
            stage_cfg=stage2_stage_cfg,
            loss_cfg=stage2_loss_lvl1,
            interface_cfg=stage2_interface,
            early_stop_cfg=stage2_early_stop,
            rollback_cfg=rollback_cfg,
            max_epochs=120,
            load_weights_from=None,
            reset_optimizer=True,
            model_updates=default_model_updates,
        ),
        # Stage 2.2 - B only, add comm_b[2]
        _phase(
            stage_cfg=stage2_stage_cfg,
            loss_cfg=stage2_loss_lvl12,
            interface_cfg=stage2_interface,
            early_stop_cfg=stage2_early_stop,
            rollback_cfg=rollback_cfg,
            max_epochs=120,
            load_weights_from=None,
            reset_optimizer=True,
            model_updates=default_model_updates,
        ),
        # Stage 2.3 - B only, add comm_b[3]
        _phase(
            stage_cfg=stage2_stage_cfg,
            loss_cfg=stage2_loss_lvl123,
            interface_cfg=stage2_interface,
            early_stop_cfg=stage2_early_stop,
            rollback_cfg=rollback_cfg,
            max_epochs=120,
            load_weights_from=None,
            reset_optimizer=True,
            model_updates=default_model_updates,
        ),
        # Stage 2.4 - B only, add shoot
        _phase(
            stage_cfg=stage2_stage_cfg,
            loss_cfg=stage2_loss_with_shoot,
            interface_cfg=stage2_interface,
            early_stop_cfg=stage2_early_stop,
            rollback_cfg=rollback_cfg,
            max_epochs=180,
            load_weights_from=None,
            reset_optimizer=True,
            model_updates=default_model_updates,
        ),
        # Stage 3 - joint AB
        _phase(
            stage_cfg=stage3_stage_cfg,
            loss_cfg=stage3_loss_cfg,
            interface_cfg=stage3_interface,
            early_stop_cfg=stage3_early_stop,
            rollback_cfg=rollback_cfg,
            max_epochs=250,
            load_weights_from=None,
            reset_optimizer=True,
            model_updates=default_model_updates,
        ),
    ]

    return phases, training_cfg


def main() -> None:
    mod = _load_training_module(TARGET_SCRIPT)
    phases, training_cfg = build_three_stage_plan()

    print("=" * 80)
    print("Running three-stage curriculum via train_multi_phase()")
    print("Stage 1 : A only")
    print("Stage 2 : B only curriculum (implemented as 4 consecutive phases)")
    print("Stage 3 : joint AB fine-tuning")
    print("=" * 80)

    result = mod.train_multi_phase(phases, training_cfg)
    mod.print_verbosity(result, level=1, config=training_cfg)
    model_cfg = {
        "N2": 16,
        "FIELD_SIZE": 4,
        "COMMS_SIZE": 1,
        "SEED": 1234,
        "P_HIGH": 1.0,
        "BETA_INPUT": 10.0,
        "ALPHA_FOR_PR_LAYERS": 0.3,
    }

    dataset_cfg = {
        "DATASET_DIR": "WIP/dataset",
        "NUM_GAMES_DATASET": 250_000,
        "BATCH": 32,
        "SEED": 1234,
        "BETA_INPUT": 10.0,
    }

    tournament_cfg = {
        "ROOT": str(ROOT),
        "FIELD_SIZE": 4,
        "COMMS_SIZE": 1,
        "GAMES_IN_EVAL_TOURNAMENT": 1000,
        "CHANNEL_NOISE": 0.0,
        "ENEMY_PROBABILITY": 0.5,
        "GAMEPLAY_ADAPTER_BETA": 10.0,
        "HARDENING_BETWEEN_LEVELS_FOR_MODEL_A": False,
        "HARDENING_BETWEEN_LEVELS_FOR_MODEL_B": False,
        "MODEL_A_WEIGHTS_IN": "WIP/checkpoints/combined_ab/model_a_latest.weights.h5",
        "MODEL_B_WEIGHTS_IN": "WIP/checkpoints/combined_ab/model_b_latest.weights.h5",
    }
    from pyr_tournament_eval_standalone import run_tournament_evaluation
    result = run_tournament_evaluation(model_cfg, tournament_cfg, dataset_cfg, verbose=True)
    print(result)
if __name__ == "__main__":
    main()
