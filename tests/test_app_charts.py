"""
Tests for the pure chart-generation helpers in app.py. These only take
plain Python/numpy/PIL data structures (no model, no dataset) and return
matplotlib Figures, so they are fully testable in isolation.
"""
import matplotlib.figure
from PIL import Image


def _make_face_result(face_num=1, color="#00FFB2"):
    return {
        "face_num": face_num,
        "color": color,
        "detect_conf": 97.5,
        "crop": Image.new("RGB", (64, 64), color=(120, 120, 120)),
        "top_names": ["alice", "bob", "carol"],
        "top_confs": [70.0, 20.0, 10.0],
        "best_name": "alice",
        "best_conf": 70.0,
    }


def test_make_confidence_chart_returns_none_for_empty_results(app_module):
    assert app_module.make_confidence_chart([]) is None
    assert app_module.make_confidence_chart(None) is None


def test_make_confidence_chart_returns_figure_for_single_face(app_module):
    fig = app_module.make_confidence_chart([_make_face_result()])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1


def test_make_confidence_chart_handles_multiple_faces(app_module):
    results = [_make_face_result(face_num=1), _make_face_result(face_num=2)]
    fig = app_module.make_confidence_chart(results)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 2


def test_make_radar_chart_returns_none_for_empty_results(app_module):
    assert app_module.make_radar_chart([]) is None


def test_make_radar_chart_returns_figure(app_module):
    fig = app_module.make_radar_chart([_make_face_result()])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1


def test_make_face_strip_returns_none_for_empty_results(app_module):
    assert app_module.make_face_strip([]) is None


def test_make_face_strip_returns_figure(app_module):
    fig = app_module.make_face_strip([_make_face_result()])
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(fig.axes) == 1
