from __future__ import annotations
from typing import Collection

from fuzzy_inference.fuzzy_set import FuzzySet
import numpy as np


class LinguisticVariable:
    def __init__(self, name, terms: dict[str, FuzzySet], range: Collection = (0, 1)) -> None:
        self.name = name
        self.terms = terms
        self._value: float | FuzzySet = 0
        self.min, self.max = range
        self.fuzzified: dict[str, float] = {}
        self.output_measure: FuzzySet = FuzzySet.uniform(0)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: int | float | FuzzySet) -> None:
        self._value = value
        if type(value) == float or type(value) == int:
            self.fuzzified = self.fuzzify_crisp(self._value)  # type: ignore
        elif type(value) == FuzzySet:
            self.fuzzified = self.fuzzify_fuzzy(self._value)  # type: ignore
        else:
            raise ValueError(str(type(value)) + ' is not supported.')


    def fuzzify_crisp(self, x: float) -> dict[str, float]:
        fuz = {}
        for k, v in self.terms.items():
            fuz[k] = v.mu(x)
        return fuz

    def fuzzify_fuzzy(self, x: FuzzySet) -> dict[str, float]:
        fuz = {}
        for k, v in self.terms.items():
            fuz[k] = FuzzySet.intersection(
                v, x, self.min, self.max).height
        return fuz
