import numpy as np
from nektsrs.grid import SimpleGrid1D, Grid1D

__all__ = ["SimpleGrid2D", "Grid2D"]


class SimpleGrid2D:
    def __init__(
        self,
        start1: float,
        end1: float,
        start2: float,
        end2: float,
        n1: int,
        n2: int,
        lx: int,
    ) -> None:

        self.start1 = start1
        self.end1 = end1
        self.start2 = start2
        self.end2 = end2
        self.n1 = n1
        self.n2 = n2
        self.lx = lx

        g1 = SimpleGrid1D(start1, end1, n1, lx)
        g2 = SimpleGrid1D(start2, end2, n2, lx)

        gllx, glly = np.meshgrid(g1.gll, g2.gll)
        self.gll = np.stack((gllx.flatten(), glly.flatten()), axis=1)


class Grid2D:
    def __init__(
        self, edges1: np.ndarray, edges2: np.ndarray, lx: int
    ) -> None:
        self.start1 = edges1[0]
        self.start2 = edges2[0]
        self.end1 = edges1[-1]
        self.end2 = edges2[-1]
        self.n1 = edges1.size - 1
        self.n2 = edges2.size - 1
        self.lx = lx

        g1 = Grid1D(edges1, lx)
        g2 = Grid1D(edges2, lx)

        gllx, glly = np.meshgrid(g1.gll, g2.gll)
        self.gll = np.stack((gllx.flatten(), glly.flatten()), axis=1)
