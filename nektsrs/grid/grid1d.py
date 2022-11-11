import numpy as np
from nektsrs.gll import gll as gll_points

__all__ = ["SimpleGrid1D", "Grid1D"]


class SimpleGrid1D:
    def __init__(self, start: float, end: float, n: int, lx: int) -> None:

        self.start = start
        self.end = end
        self.n = n
        self.lx = lx

        self.edges = np.linspace(start, end, n + 1)

        points, _ = gll_points(lx)

        gll = np.zeros((n, lx))
        for i in range(n):
            e_start = self.edges[i]
            e_end = self.edges[i + 1]
            gll[i, :] = (points + 1) * (e_end - e_start) / 2 + e_start

        gll = np.unique(gll.flatten())
        self.gll = gll


class Grid1D:
    def __init__(self, edges: np.ndarray, lx: int) -> None:

        self.start = edges[0]
        self.end = edges[-1]
        self.n = edges.size - 1
        self.lx = lx
        self.edges = edges

        points, _ = gll_points(lx)

        gll = np.zeros((self.n, lx))
        for i in range(self.n):
            e_start = self.edges[i]
            e_end = self.edges[i + 1]
            gll[i, :] = (points + 1) * (e_end - e_start) / 2 + e_start

        gll = np.unique(gll.flatten())
        self.gll = gll
