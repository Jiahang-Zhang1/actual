from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.backends.base import BackendOutput, ModelBackend
from app.compat import apply_sklearn_compat_patches


def _softmax_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(matrix)
    denom = np.sum(exp_matrix, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp_matrix / denom


def _ensure_2d(matrix: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        if n_classes == 2:
            matrix = np.stack([-matrix, matrix], axis=1)
        else:
            matrix = matrix.reshape(1, -1)
    return matrix


def load_sklearn_model(model_path: str):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing sklearn model artifact: {path}")

    apply_sklearn_compat_patches()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(path)

    if not hasattr(model, "transform_input"):
        try:
            model.transform_input = None
        except Exception:
            pass
    return model


def extract_classes(model) -> list[str]:
    if hasattr(model, "classes_"):
        return [str(x) for x in model.classes_]

    named_steps = getattr(model, "named_steps", {})
    clf = named_steps.get("clf")
    if clf is not None and hasattr(clf, "classes_"):
        return [str(x) for x in clf.classes_]

    raise RuntimeError("Could not determine classes_ from the sklearn artifact.")


class SklearnBackend(ModelBackend):
    kind = "baseline"

    def __init__(self, model_path: str):
        self.model_path = str(Path(model_path))
        self.model = load_sklearn_model(self.model_path)
        self.classes = extract_classes(self.model)

    def _predict_matrix(self, frame: pd.DataFrame) -> tuple[np.ndarray, bool]:
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(frame), dtype=float), False

        if hasattr(self.model, "decision_function"):
            return np.asarray(self.model.decision_function(frame), dtype=float), True

        named_steps = getattr(self.model, "named_steps", {})
        clf = named_steps.get("clf")
        preprocessor = named_steps.get("preprocessor")
        if clf is None:
            raise RuntimeError("The sklearn artifact does not expose a classifier step.")

        transformed = preprocessor.transform(frame) if preprocessor is not None else frame
        if hasattr(clf, "predict_proba"):
            return np.asarray(clf.predict_proba(transformed), dtype=float), False
        if hasattr(clf, "decision_function"):
            return np.asarray(clf.decision_function(transformed), dtype=float), True

        raise RuntimeError("The sklearn artifact exposes neither predict_proba nor decision_function.")

    def predict(self, frame: pd.DataFrame) -> BackendOutput:
        labels = [str(x) for x in self.model.predict(frame)]
        matrix, raw_scores = self._predict_matrix(frame)
        matrix = _ensure_2d(matrix, len(self.classes))

        if raw_scores:
            probabilities = _softmax_rows(matrix)
        else:
            row_sums = np.sum(matrix, axis=1)
            looks_like_probs = bool(np.all(matrix >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-4))
            probabilities = matrix if looks_like_probs else _softmax_rows(matrix)

        return BackendOutput(labels=labels, probabilities=probabilities, classes=self.classes)

    def providers(self) -> list[str]:
        return ["sklearn-cpu"]
