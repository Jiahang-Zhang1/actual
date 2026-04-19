from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.backends.base import BackendOutput, ModelBackend
from app.backends.sklearn_backend import SklearnBackend
from app.feature_adapter import dataframe_to_onnx_inputs


def _softmax_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_matrix = np.exp(matrix)
    denom = np.sum(exp_matrix, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return exp_matrix / denom


class OnnxBackend(ModelBackend):
    kind = "onnx"

    def __init__(
        self,
        model_path: str,
        source_model_path: str,
        intra_op_num_threads: int = 1,
        inter_op_num_threads: int = 1,
    ):
        self.model_path = str(Path(model_path))
        self.source_model_path = str(Path(source_model_path))
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Missing ONNX model artifact: {self.model_path}")

        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.inter_op_num_threads = inter_op_num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        self.classes = SklearnBackend(self.source_model_path).classes

    def _parse_label_output(self, output: Any, fallback_probabilities: np.ndarray) -> list[str]:
        if output is None:
            best_idx = np.argmax(fallback_probabilities, axis=1)
            return [self.classes[int(i)] for i in best_idx]

        arr = np.asarray(output).reshape(-1)
        labels = []
        for item in arr:
            if isinstance(item, bytes):
                labels.append(item.decode("utf-8"))
            else:
                text = str(item)
                if text.isdigit():
                    labels.append(self.classes[int(text)])
                else:
                    labels.append(text)
        return labels

    def _parse_score_output(self, output: Any, row_count: int) -> np.ndarray:
        if output is None:
            return np.zeros((row_count, len(self.classes)), dtype=float)

        if isinstance(output, list) and output and isinstance(output[0], dict):
            matrix = np.zeros((len(output), len(self.classes)), dtype=float)
            class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
            for row_idx, row in enumerate(output):
                for key, value in row.items():
                    key_str = key.decode("utf-8") if isinstance(key, bytes) else str(key)
                    if key_str in class_to_idx:
                        matrix[row_idx, class_to_idx[key_str]] = float(value)
                    elif key_str.isdigit():
                        matrix[row_idx, int(key_str)] = float(value)
            return matrix

        arr = np.asarray(output, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr

    def predict(self, frame: pd.DataFrame) -> BackendOutput:
        feed = dataframe_to_onnx_inputs(frame)
        outputs = self.session.run(None, feed)
        output_map = {name: value for name, value in zip(self.output_names, outputs)}

        label_output = None
        score_output = None

        for name, value in output_map.items():
            lowered = name.lower()
            arr = np.asarray(value)
            if ("label" in lowered or arr.dtype.kind in {"U", "S", "O"}) and label_output is None:
                label_output = value
            elif score_output is None:
                score_output = value

        if score_output is None and outputs:
            # If only one output exists, assume it is numeric scores.
            only = outputs[0]
            try:
                arr = np.asarray(only)
                if arr.dtype.kind not in {"U", "S", "O"}:
                    score_output = only
            except Exception:
                score_output = None

        raw_matrix = self._parse_score_output(score_output, len(frame))
        row_sums = np.sum(raw_matrix, axis=1)
        looks_like_probs = bool(np.all(raw_matrix >= 0.0) and np.allclose(row_sums, 1.0, atol=1e-4))
        probabilities = raw_matrix if looks_like_probs else _softmax_rows(raw_matrix)
        labels = self._parse_label_output(label_output, probabilities)

        return BackendOutput(labels=labels, probabilities=probabilities, classes=self.classes)

    def providers(self) -> list[str]:
        return list(self.session.get_providers())
