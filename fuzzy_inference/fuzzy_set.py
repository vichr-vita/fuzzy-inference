from __future__ import annotations

from math import nan, inf
from typing import Callable
import numpy as np


def piecewise_linear(x: float, vertices: tuple[tuple]) -> float:
    """
    vertices have to be sorted by x
    memberships have to start with 0
    memberships have to end with 0
    """
    return np.interp([x], [t[0] for t in vertices], [t[1] for t in vertices])[0]


def trapezoidal(x: float, a: float, b: float, c: float, d: float, height: float = 1) -> float:
    return piecewise_linear(x, (
        (a, 0),
        (b, height),
        (c, height),
        (d, 0)
    ))  # type: ignore


def triangular(x: float, a: float, b: float, c: float, height: float = 1.0) -> float:
    return piecewise_linear(x, (
        (a, 0),
        (b, height),
        (c, 0)
    ))  # type: ignore


def l_ramp(x: float, start: float, end: float, height: float = 1) -> float:
    return piecewise_linear(x, (
        (-np.Inf, height),
        (start, height),
        (end, 0)
    ))  # type: ignore


def r_ramp(x: float, start: float, end: float, height: float = 1) -> float:
    return piecewise_linear(x, (
        (start, 0),
        (end, height),
        (np.Inf, height)
    ))  # type: ignore


class FuzzySet:

    def __init__(self, mu_x: Callable, **kwargs) -> None:
        self.mu_x = mu_x
        self.kwargs = kwargs

    def mu(self, x: float) -> float:
        return self.mu_x(x, **self.kwargs)

    def discrete(self, x_from: float, x_to: float, resolution: int):
        return np.array([np.array([x, self.mu(x)]) for x in np.linspace(x_from, x_to, resolution)])

    @staticmethod
    def uniform(height: float = 1):
        return FuzzySet(lambda x: height)

    @staticmethod
    def l_ramp(start: float, end: float, height: float = 1):
        return FuzzySet(l_ramp, start=start, end=end, height=height)

    @staticmethod
    def r_ramp(start: float, end: float, height: float = 1):
        return FuzzySet(r_ramp, start=start, end=end, height=height)

    @staticmethod
    def piecewise_linear(vertices: tuple) -> FuzzySet:
        return FuzzySet(piecewise_linear, vertices=vertices)

    @staticmethod
    def trapezoidal(a: float, b: float, c: float, d: float, height: float = 1) -> FuzzySet:
        return FuzzySet(trapezoidal, a=a, b=b, c=c, d=d, height=height)

    @staticmethod
    def triangular(a: float, b: float, c: float, height: float = 1) -> FuzzySet:
        return FuzzySet(triangular, a=a, b=b, c=c, height=height)

    def intersection(self, f2: FuzzySet, x_from, x_to, resolution=100):
        """
        using minimum t-norm
        SOMEDAY: different t-norms
        """
        x = np.linspace(x_from, x_to, resolution)
        return np.vstack((x, np.minimum(self.discrete(x_from, x_to, resolution)[
            :, 1], f2.discrete(x_from, x_to, resolution)[:, 1]))).T

    def union(self, f2: FuzzySet, x_from, x_to, resolution=100):
        """
        using minimum t-norm
        SOMEDAY: different t-norms
        """
        x = np.linspace(x_from, x_to, resolution)
        return np.vstack((x, np.maximum(self.discrete(x_from, x_to, resolution)[
            :, 1], f2.discrete(x_from, x_to, resolution)[:, 1]))).T
