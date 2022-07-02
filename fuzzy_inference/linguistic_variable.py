from __future__ import annotations
from typing import Collection

from fuzzy_inference.fuzzy_set import FuzzySet
import numpy as np


class LinguisticVariable:
    def __init__(self, name, terms: dict[str, FuzzySet], range: Collection = (0, 1)) -> None:
        self.name = name
        self.terms = terms
        self._value = None
        self.min, self.max = range

    def value(self, value: float | FuzzySet) -> None:
        self._value = value

    def fuzzify_crisp(self, x: float) -> dict[str, float]:
        fuz = {}
        for k, v in self.terms.items():
            fuz[k] = v.mu(x)
        return fuz

    def fuzzify_fuzzy(self, x: FuzzySet) -> dict[str, float]:
        fuz = {}
        for k, v in self.terms.items():
            fuz[k] = np.max(v.intersection(x, self.min, self.max)[:, 1])
        return fuz
