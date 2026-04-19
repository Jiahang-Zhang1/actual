from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import numpy as np


@dataclass
class BackendOutput:
    labels: list[str]
    probabilities: np.ndarray
    classes: list[str]


class ModelBackend(ABC):
    kind: str = "unknown"

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> BackendOutput:
        raise NotImplementedError

    @abstractmethod
    def providers(self) -> list[str]:
        raise NotImplementedError
