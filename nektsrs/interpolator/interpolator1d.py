import numpy as np
from scipy.interpolate import lagrange
from nektsrs.grid import Grid1D, SimpleGrid1D
from typing import Union, Dict
from nektsrs.gll import gll


__all__ = ["Interpolator1D"]


class Interpolator1D:
    def __init__(self, grid: Union[Grid1D, SimpleGrid1D]) -> None:
        self.grid = grid
        self.ref_gll, _ = gll(grid.lx)

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
        if i < 0 or i > self.nelems - 1:
            raise ValueError(f"Element index {i} is out of bounds.")

        return self.edges[i], self.edges[i + 1]

    def element_gll_points(self, i: int):
        """Get gll point of a particular element by its index."""

        if i < 0 or i > self.nelems - 1:
            raise ValueError(f"Element index {i} is out of bounds.")

        ind = self.element_gll_indices(i)
        return self.gll[ind[0] : ind[1]]

    def element_gll_indices(self, i: int):
        if i < 0 or i > self.nelems - 1:
            raise ValueError(f"Element index {i} is out of bounds.")

        npoly = self.lx - 1
        return i * npoly, i * npoly + self.lx

    def data_element_stats(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """Data for normalizing the data within each element."""
        means = np.zeros(self.nelems)
        stds = np.zeros(self.nelems)
        for i in range(self.nelems):
            ind = self.element_gll_indices(i)
            means[i] = np.mean(data[ind[0] : ind[1]])
            stdi = np.std(data[ind[0] : ind[1]])
            stds[i] = stdi if stdi != 0 else 1.0
        return means, stds

    def build_polys(
        self,
        data: np.ndarray,
        data_means: np.ndarray,
        data_stds: np.ndarray,
        element_ind: np.ndarray,
    ) -> Dict:
        """Build Lagrange polynomials for elements of given indices.

        The interpolation is performed on a reference [-1, 1] element.
        The data is demeaned and divided by the standard deviation.
        This transformation has to be reversed later.
        This normalization is necessary for stability of lagrangian
        interpolation.

        """
        polys = dict()
        for i, eli in enumerate(element_ind):
            ind = self.element_gll_indices(eli)
            polys[eli] = lagrange(
                self.ref_gll,
                (data[ind[0] : ind[1]] - data_means[eli]) / data_stds[eli],
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

        if np.max(points) > np.max(self.gll) or np.min(points) < np.min(
            self.gll
        ):
            raise ValueError("Interpolation point out of bound of the mesh.")

        # Searches for element index where the points will lie
        element_ind = np.searchsorted(self.edges, points, side="left") - 1

        for i, eli in enumerate(element_ind):
            if eli < 0:
                element_ind[i] = 0

        data_means, data_stds = self.data_element_stats(data)
        polys = self.build_polys(
            data, data_means, data_stds, np.unique(element_ind)
        )

        for i, eli in enumerate(element_ind):
            edges = self.element_edges(eli)
            ref_p = (points[i] - edges[0]) / (edges[1] - edges[0]) * 2 - 1
            values[i] = polys[eli](ref_p) * data_stds[eli] + data_means[eli]

        return values
