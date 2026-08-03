"""
Shared pytest fixtures.

`app.py` loads the trained SVM classifier, the label map, MTCNN and the
FaceNet backbone at *import time* (module-level code, not behind an
`if __name__ == "__main__":` guard). None of those artifacts are meant to
live in this repo (see README -> "Reproducing the model"), so the test
suite never touches the real files or downloads real weights: this
fixture patches `joblib.load`, `numpy.load` and the `facenet_pytorch`
classes with lightweight fakes *before* `app` is imported, then imports
it fresh. Individual tests can further override `app.mtcnn` / `app.resnet`
/ `app.model` / `app.idx2label` to control behaviour precisely.
"""
import sys
from unittest.mock import MagicMock

import numpy as np
import joblib
import facenet_pytorch
import pytest


DUMMY_LABELS = np.array(["alice", "bob", "carol"])


class DummyClassifier:
    """Stand-in for the trained SVM pipeline (StandardScaler + SVC)."""

    n_classes = len(DUMMY_LABELS)

    def predict_proba(self, embeddings):
        n = np.asarray(embeddings).shape[0]
        # Deterministic, descending "confidence" per row so tests can
        # assert on top-K ordering without randomness.
        base = np.array([0.7, 0.2, 0.1])
        return np.tile(base, (n, 1))


def make_fake_embedding_tensor(array):
    """A stand-in for a torch tensor's `.cpu().numpy()` chain."""
    fake = MagicMock(name="fake_torch_tensor")
    fake.cpu.return_value.numpy.return_value = np.asarray(array, dtype="float32")
    return fake


def _make_resnet_ctor_mock():
    """Mocks `InceptionResnetV1(pretrained=...).eval().to(device)` at
    import time only. Import-time behaviour just needs to not crash;
    tests that exercise real inference override `app.resnet` directly.
    """
    resnet_instance = MagicMock(name="resnet_instance")
    resnet_instance.parameters.return_value = []  # `for p in resnet.parameters()`

    eval_result = MagicMock(name="eval_result")
    eval_result.to.return_value = resnet_instance

    ctor_result = MagicMock(name="ctor_result")
    ctor_result.eval.return_value = eval_result

    return MagicMock(name="InceptionResnetV1_cls", return_value=ctor_result)


@pytest.fixture
def app_module(monkeypatch):
    """Import `app` with every heavyweight/model-loading call mocked out."""
    monkeypatch.setattr(joblib, "load", lambda path: DummyClassifier())
    monkeypatch.setattr(
        np, "load", lambda path, allow_pickle=False: DUMMY_LABELS
    )
    monkeypatch.setattr(facenet_pytorch, "MTCNN", MagicMock(name="MTCNN_cls"))
    monkeypatch.setattr(
        facenet_pytorch, "InceptionResnetV1", _make_resnet_ctor_mock()
    )

    sys.modules.pop("app", None)
    import app  # noqa: E402  (imported late on purpose, after patching)

    yield app

    sys.modules.pop("app", None)
