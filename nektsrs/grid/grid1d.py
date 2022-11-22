import numpy as np
from nektsrs.gll import gll as gll_points

__all__ = ["SimpleGrid1D", "Grid1D"]


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

    def element_edges(self, i: int):
        """Get the edges of a particular element.

        """
        if i < 0 or i > self.n - 1:
            raise ValueError(f"Element index {i} is out of bounds.")

        return self.edges[i], self.edges[i + 1]

    def element_gll_points(self, i: int):
        """Get gll point of a particular element by its index.

        """
        if i < 0 or i > self.n - 1:
            raise ValueError(f"Element index {i} is out of bounds.")

        ind = self.element_gll_indices(i)
        return self.gll[ind[0] : ind[1]]

    def element_gll_indices(self, i: int):
        """Get the indces of the gll point of a particular element.

        """
        if i < 0 or i > self.n - 1:
            raise ValueError(f"Element index {i} is out of bounds.")

        npoly = self.lx - 1
        return i * npoly, i * npoly + self.lx


class SimpleGrid1D(Grid1D):
    def __init__(self, start: float, end: float, n: int, lx: int) -> None:
        edges = np.linspace(start, end, n + 1)
        Grid1D.__init__(self, edges=edges, lx=lx)
