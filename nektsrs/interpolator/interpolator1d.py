import numpy as np
from scipy.interpolate import lagrange
from nektsrs.grid import Grid1D, SimpleGrid1D
from typing import Union, Dict


__all__ = ["Interpolator1D"]


class Interpolator1D:
    def __init__(self, grid: Union[Grid1D, SimpleGrid1D]) -> None:
        self.grid = grid

    @property
    def edges(self):
        return self.grid.edges

    @property
    def gll(self):
        return self.grid.gll

    @property
    def lx(self):
        return self.grid.lx

    @property
    def nelems(self):
        return self.grid.n

    def element_edges(self, i: int):
        return self.edges[i], self.edges[i + 1]

    def element_gll_points(self, i: int):
        """Get gll point of a particular element by its index."""
        ind = self.element_gll_indices(i)
        return self.gll[ind[0] : ind[1]]

    def element_gll_indices(self, i):
        npoly = self.lx - 1
        return i * npoly, i * npoly + self.lx

    def build_polys(self, data: np.ndarray, element_ind: np.ndarray) -> Dict:
        """Build Lagrange polynomials for elements of given indices."""

        polys = dict()
        for i, eli in enumerate(element_ind):
            ind = self.element_gll_indices(eli)
            polys[eli] = lagrange(
                self.element_gll_points(eli), data[ind[0] : ind[1]]
            )
        return polys

    def interpolate(
        self, data: np.ndarray, points: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Interpolate to new data points given data on the grid.

        Parameters
        ----------
            data: 1d nd.array of size self.gll
                The data at the points of the given grid, for several
                fields.

            points: 1d nd.array
                The points where we want the interpolated values.

        """

        if data.size != self.gll.size:
            raise ValueError("Data is of incorrect size!")

        if type(points) is float:
            points = points * np.ones(1)
        values = np.zeros(points.shape[0])

        # Searches for element index where the points will lie
        element_ind = np.searchsorted(self.edges, points) - 1

        polys = self.build_polys(data, np.unique(element_ind))

        for i, eli in enumerate(element_ind):
            values[i] = polys[eli](points[i])

        return values
