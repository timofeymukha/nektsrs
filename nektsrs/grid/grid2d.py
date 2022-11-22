import numpy as np
from nektsrs.grid import Grid1D

__all__ = ["SimpleGrid2D", "Grid2D"]


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
        self.edges1 = edges1
        self.edges2 = edges2

        g1 = Grid1D(edges1, lx)
        g2 = Grid1D(edges2, lx)

        self.gll1 = g1.gll
        self.gll2 = g2.gll

        gllx, glly = np.meshgrid(g1.gll, g2.gll)
        self.gll = np.stack((gllx.flatten(), glly.flatten()), axis=1)

    def element_edges(self, i: int, j: int):
        """Get the edges of a particular element."""
        if i < 0 or i > self.n1 - 1 or j < 0 or j > self.n2 - 1:
            raise ValueError(f"Element index {i}, {j} is out of bounds.")

        return (
            self.edges1[i],
            self.edges1[i + 1],
            self.edges2[j],
            self.edges2[j + 1],
        )

    def element_gll_indices(self, i: int, j: int):
        """Get the indces of the gll point of a particular element."""
        if i < 0 or i > self.n2 - 1 or j < 0 or j > self.n2 - 1:
            raise ValueError(f"Element index {i}, {j} is out of bounds.")

        npoly = self.lx - 1
        return (i * npoly, i * npoly + self.lx, j * npoly, j * npoly + self.lx)

    def element_gll_points(self, i: int, j: int):
        """Get gll point of a particular element by its index."""
        if i < 0 or i > self.n2 - 1 or j < 0 or j > self.n2 - 1:
            raise ValueError(f"Element index {i}, {j} is out of bounds.")

        ind = self.element_gll_indices(i, j)
        return self.gll1[ind[0] : ind[1]], self.gll2[ind[2] : ind[3]]


class SimpleGrid2D(Grid2D):
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

        edges1 = np.linspace(start1, end1, n1 + 1)
        edges2 = np.linspace(start2, end2, n2 + 1)
        Grid2D.__init__(self, edges1, edges2, lx)
