from pathlib import Path

import sys

# tests/ -> WIP/
WIP_DIR = Path(__file__).resolve().parent.parent
if str(WIP_DIR) not in sys.path:
    sys.path.insert(0, str(WIP_DIR))

from config import get_config

def test_get_config_has_required_keys():
    cfg = get_config()
    required = [
        "ROOT", "WIP", "DATASET_DIR", "CHECKPOINT_DIR", "LOG_DIR",
        "SEED", "BATCH", "EPOCHS", "LR",
        "LOAD_WEIGHTS_ON_START", "MODEL_A_WEIGHTS_IN", "MODEL_B_WEIGHTS_IN",
        "SAVE_WEIGHTS_EVERY", "SAVE_WEIGHTS_AT_END", "LOG_FLUSH_EVERY_EPOCHS",
        "weights",
    ]
    for k in required:
        assert k in cfg, f"Missing key: {k}"


def test_paths_are_path_objects():
    cfg = get_config()
    for k in ["ROOT", "WIP", "DATASET_DIR", "CHECKPOINT_DIR", "LOG_DIR"]:
        assert isinstance(cfg[k], Path)