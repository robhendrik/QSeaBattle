from __future__ import annotations

from pathlib import Path
from typing import Any
from pathlib import Path
from typing import Any


def get_config() -> dict[str, Any]:
    ROOT = Path(__file__).resolve().parent.parent   # WIP/config.py -> ROOT
    WIP = ROOT / "WIP"

    # Core dataset generation knobs (needed by data_build.py)
    N2 = 16
    FIELD_SIZE = 4
    COMMS_SIZE = 1
    DEPTH = 4                  # keep explicit to match existing script behavior
    NUM_GAMES_DATASET = 10_000
    BETA_INPUT = 10.0

    # Paths
    DATASET_DIR = WIP / "dataset_10k"
    CHECKPOINT_DIR = WIP / "checkpoints" / "combined_ab"
    LOG_DIR = WIP / "logs"

    # Training knobs
    SEED = 42
    BATCH = 256
    EPOCHS = 1000
    LR = 1e-3

    # Weight load/save
    LOAD_WEIGHTS_ON_START = True
    MODEL_A_WEIGHTS_IN = "checkpoints\\combined_ab\\model_a_latest.weights.h5"
    MODEL_B_WEIGHTS_IN = "checkpoints\\combined_ab\\model_b_latest.weights.h5"
    SAVE_WEIGHTS_EVERY = 50
    SAVE_WEIGHTS_AT_END = True

    # Logging
    LOG_FLUSH_EVERY_EPOCHS = 0

    weights = {
        "TRAINING_MODE": "combined",
        # ...existing code...
        # keep your existing weight keys/values unchanged
    }
    ENABLE_ROLLBACK = False
    P_HIGH = 1.0
    return {
        "ROOT": ROOT,
        "WIP": WIP,
        "N2": N2,
        "FIELD_SIZE": FIELD_SIZE,
        "COMMS_SIZE": COMMS_SIZE,
        "DEPTH": DEPTH,
        "P_HIGH": P_HIGH,
        "NUM_GAMES_DATASET": NUM_GAMES_DATASET,
        "BETA_INPUT": BETA_INPUT,
        "DATASET_DIR": DATASET_DIR,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "LOG_DIR": LOG_DIR,
        "SEED": SEED,
        "BATCH": BATCH,
        "EPOCHS": EPOCHS,
        "LR": LR,
        "LOAD_WEIGHTS_ON_START": LOAD_WEIGHTS_ON_START,
        "MODEL_A_WEIGHTS_IN": MODEL_A_WEIGHTS_IN,
        "MODEL_B_WEIGHTS_IN": MODEL_B_WEIGHTS_IN,
        "SAVE_WEIGHTS_EVERY": SAVE_WEIGHTS_EVERY,
        "SAVE_WEIGHTS_AT_END": SAVE_WEIGHTS_AT_END,
        "LOG_FLUSH_EVERY_EPOCHS": LOG_FLUSH_EVERY_EPOCHS,
        "ENABLE_ROLLBACK": ENABLE_ROLLBACK,
        "weights": weights,
    }

