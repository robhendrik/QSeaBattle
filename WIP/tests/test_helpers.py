from pathlib import Path
import tensorflow as tf
import sys

# tests/ -> WIP/
WIP_DIR = Path(__file__).resolve().parent.parent
if str(WIP_DIR) not in sys.path:
    sys.path.insert(0, str(WIP_DIR))

from helpers import (
    _wfile,
    bce_from_logits,
    harden_ste,
    logits_l2_reg,
    save_ab_weights,
    load_ab_weights,
    init_run_log,
    append_epoch_log,
    flush_run_log,
)


class DummyModel:
    def __init__(self, built=True):
        self.built = built
        self.saved = []
        self.loaded = []

    def save_weights_to(self, p):
        Path(p).write_bytes(b"ok")
        self.saved.append(p)

    def load_weights_from(self, p):
        assert Path(p).exists()
        self.loaded.append(p)


def test_wfile():
    p = _wfile(Path("x"), "model_a", "latest")
    assert str(p).endswith("model_a_latest.weights.h5")


def test_save_load_weights(tmp_path):
    a = DummyModel(built=True)
    b = DummyModel(built=True)

    a_path, b_path = save_ab_weights(a, b, tmp_path, "epoch_0001")
    assert a_path is not None and b_path is not None
    assert a_path.exists() and b_path.exists()

    ok = load_ab_weights(a, b, a_path, b_path)
    assert ok is True


def test_save_weights_skips_when_unbuilt(tmp_path):
    a = DummyModel(built=False)
    b = DummyModel(built=True)
    a_path, b_path = save_ab_weights(a, b, tmp_path, "epoch_0001")
    assert a_path is None and b_path is None


def test_logging_roundtrip(tmp_path):
    run_log, log_path = init_run_log(tmp_path, {"start_mode": "fresh", "loaded_from": None})
    append_epoch_log(run_log, {"epoch": 0, "total": 1.23})
    flush_run_log(run_log, log_path)
    assert log_path.exists()
    assert log_path.stat().st_size > 0


def test_loss_helpers_basic():
    tgt = tf.constant([-1.0, 2.0, 0.0], dtype=tf.float32)
    pred = tf.constant([0.1, -0.2, 0.3], dtype=tf.float32)

    loss = bce_from_logits(tgt, pred)
    assert loss.shape == ()
    assert float(loss.numpy()) >= 0.0

    x = tf.constant([-0.2, 0.0, 0.7], dtype=tf.float32)
    y = harden_ste(x, beta=5.0)
    assert y.numpy().tolist() == [-5.0, 5.0, 5.0]

    reg = logits_l2_reg(tf.constant([1.0, -1.0]), tf.constant([2.0]), weight=1e-4)
    assert reg.shape == ()
    assert float(reg.numpy()) > 0.0