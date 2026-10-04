"""
Tests for faces.py's matching and persistence logic. `match_face` is tested
against the real `face_recognition.compare_faces`/`face_distance` functions
using small synthetic encodings (128-d vectors, matching face_recognition's
real output shape) rather than actual photos — this exercises the matching
and tie-breaking logic without needing a real face image or dlib's detector.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentryvision import config, faces  # noqa: E402


def _fake_encoding(seed):
    rng = np.random.RandomState(seed)
    return rng.rand(128)


def test_match_face_returns_unknown_with_no_known_faces():
    result = faces.match_face(_fake_encoding(0), known_names=[], known_encodings=[])
    assert result == "Unknown"


def test_match_face_finds_exact_match():
    known_encoding = _fake_encoding(1)
    result = faces.match_face(
        known_encoding, known_names=["Mazidul"], known_encodings=[known_encoding]
    )
    assert result == "Mazidul"


def test_match_face_returns_unknown_for_dissimilar_encoding():
    known_encoding = _fake_encoding(1)
    # A very different random vector is extremely unlikely to fall within
    # face_recognition's default distance tolerance of a specific known face.
    unrelated_encoding = _fake_encoding(99) * 10
    result = faces.match_face(
        unrelated_encoding, known_names=["Mazidul"], known_encodings=[known_encoding]
    )
    assert result == "Unknown"


def test_load_known_encodings_empty_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENCODINGS_PATH", str(tmp_path / "missing.pkl"))
    assert faces.load_known_encodings() == {}


def test_save_and_load_known_encodings_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENCODINGS_PATH", str(tmp_path / "encodings.pkl"))
    original = {"Mazidul": _fake_encoding(1), "Rahim": _fake_encoding(2)}

    faces.save_known_encodings(original)
    loaded = faces.load_known_encodings()

    assert set(loaded.keys()) == {"Mazidul", "Rahim"}
    assert np.allclose(loaded["Mazidul"], original["Mazidul"])
