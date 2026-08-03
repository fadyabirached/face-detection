"""
Tests for the detection/classification control-flow in app.py
(`run_recognition`, `process_image`). The FaceNet/MTCNN/SVM objects are
replaced with lightweight fakes (see conftest.py) so no trained model
file, dataset, or downloaded weights are ever needed.
"""
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image


def make_fake_embedding_tensor(array):
    """A stand-in for a torch tensor's `.cpu().numpy()` chain."""
    fake = MagicMock(name="fake_torch_tensor")
    fake.cpu.return_value.numpy.return_value = np.asarray(array, dtype="float32")
    return fake


def _blank_image(size=(120, 120)):
    return Image.new("RGB", size, color=(200, 200, 200))


def test_run_recognition_with_no_image_returns_early(app_module):
    annotated, results, status = app_module.run_recognition(None)
    assert annotated is None
    assert results is None
    assert "No image provided" in status


def test_run_recognition_with_no_faces_detected(app_module):
    app_module.mtcnn.detect = MagicMock(return_value=(None, None))

    img = _blank_image()
    annotated, results, status = app_module.run_recognition(img)

    assert results is None
    assert "No faces detected" in status
    # Original image is returned untouched when nothing is found.
    assert annotated.size == img.size


def test_run_recognition_filters_low_confidence_detections(app_module):
    # Below config.DETECT_THRESH (0.85)
    app_module.mtcnn.detect = MagicMock(
        return_value=(np.array([[10, 10, 50, 50]]), np.array([0.50]))
    )

    img = _blank_image()
    annotated, results, status = app_module.run_recognition(img)

    assert results is None
    assert "confidence too low" in status


def test_run_recognition_happy_path_classifies_detected_face(app_module):
    # One confidently-detected face box.
    app_module.mtcnn.detect = MagicMock(
        return_value=(
            np.array([[10, 10, 90, 90]], dtype="float32"),
            np.array([0.99], dtype="float32"),
        )
    )
    # `mtcnn(face_crop)` (embedding-extraction call) returns a fake
    # aligned face tensor — shape (3, 160, 160) like the real MTCNN.
    app_module.mtcnn.return_value = torch.zeros(3, 160, 160)
    # `resnet(face_tensor)` returns a fake 512-dim embedding.
    app_module.resnet = MagicMock(
        return_value=make_fake_embedding_tensor(np.zeros((1, 512)))
    )

    img = _blank_image()
    annotated, results, status = app_module.run_recognition(img, top_k=2)

    assert "1 face(s) detected" in status
    assert len(results) == 1
    face = results[0]
    assert face["face_num"] == 1
    assert face["best_name"] == "alice"  # DummyClassifier's top class
    assert face["best_conf"] == 70.0
    assert len(face["top_names"]) == 2  # top_k=2 respected
    assert annotated.size == img.size


def test_process_image_with_no_image_returns_placeholder_tuple(app_module):
    result = app_module.process_image(None, top_k_slider=6)
    assert result[0] is None
    assert result[1] is None
    assert result[2] is None
    assert result[3] is None
    assert "Please upload an image" in result[4]


def test_process_image_with_no_faces_skips_charts(app_module):
    app_module.mtcnn.detect = MagicMock(return_value=(None, None))

    np_img = np.array(_blank_image())
    annotated, conf_fig, radar_fig, strip_fig, status = app_module.process_image(
        np_img, top_k_slider=6
    )

    assert conf_fig is None
    assert radar_fig is None
    assert strip_fig is None
    assert "No faces detected" in status
