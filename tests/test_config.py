"""
Tests for config.py — pure settings/defaults, no model files or dataset
required. `config.py` reads values via `os.getenv(...)` at import time,
so we reload the module under a patched environment to check overrides.
"""
import importlib
import sys

import pytest


@pytest.fixture
def fresh_config(monkeypatch):
    """Reload config.py after optionally patching environment variables."""
    def _load(env=None):
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)
        sys.modules.pop("config", None)
        return importlib.import_module("config")

    yield _load
    sys.modules.pop("config", None)


def test_default_paths(fresh_config):
    cfg = fresh_config()
    assert cfg.MODEL_PATH == "face_recognition_model.pkl"
    assert cfg.LABELS_PATH == "idx2label.npy"


def test_default_server_settings(fresh_config):
    cfg = fresh_config()
    assert cfg.SERVER_NAME == "0.0.0.0"
    assert cfg.SERVER_PORT == 7860
    assert cfg.SHARE is False


def test_default_model_hyperparameters(fresh_config):
    cfg = fresh_config()
    assert cfg.IMG_SIZE == 160
    assert cfg.EMBEDDING_DIM == 512
    assert cfg.DETECT_THRESH == pytest.approx(0.85)
    assert cfg.DEFAULT_TOP_K == 6
    assert cfg.MAX_TOP_K == 10
    assert cfg.DEFAULT_TOP_K <= cfg.MAX_TOP_K


def test_env_overrides_model_and_labels_path(fresh_config):
    cfg = fresh_config({
        "MODEL_PATH": "/tmp/custom_model.pkl",
        "LABELS_PATH": "/tmp/custom_labels.npy",
    })
    assert cfg.MODEL_PATH == "/tmp/custom_model.pkl"
    assert cfg.LABELS_PATH == "/tmp/custom_labels.npy"


def test_env_overrides_server_port(fresh_config):
    cfg = fresh_config({"SERVER_PORT": "9999"})
    assert cfg.SERVER_PORT == 9999
    assert isinstance(cfg.SERVER_PORT, int)


@pytest.mark.parametrize(
    "raw_value, expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("", False),
    ],
)
def test_env_override_share_is_case_insensitive_bool(fresh_config, raw_value, expected):
    cfg = fresh_config({"SHARE": raw_value})
    assert cfg.SHARE is expected


def test_device_is_cpu(fresh_config):
    # The app is pinned to CPU inference regardless of CUDA availability.
    cfg = fresh_config()
    assert cfg.DEVICE == "cpu"


def test_face_colors_nonempty_and_hex(fresh_config):
    cfg = fresh_config()
    assert len(cfg.FACE_COLORS) > 0
    for color in cfg.FACE_COLORS:
        assert color.startswith("#")
        assert len(color) == 7
