import numpy as np
from nektsrs.grid import Grid2D, SimpleGrid2D
from typing import Union
from nektsrs.gll import gll
from scipy.interpolate import BarycentricInterpolator


__all__ = ["Interpolator2D"]


class Interpolator2D:
    def __init__(self, grid: Union[Grid2D, SimpleGrid2D]) -> None:
        self.grid = grid
        self.ref_gll, _ = gll(grid.lx)

        self.interpolator = BarycentricInterpolator(xi=self.ref_gll)

    @property
    def edges1(self):
        return self.grid.edges1

    @property
    def edges2(self):
        return self.grid.edges2

    @property
    def gll1(self):
        return self.grid.gll1

    @property
    def gll2(self):
        return self.grid.gll2

    @property
    def gll(self):
        return self.grid.gll

    @property
    def lx(self):
        return self.grid.lx

    @property
    def nelems1(self):
        return self.grid.n1

    @property
    def nelems2(self):
        return self.grid.n2

    def data_element_stats(self, data: np.ndarray) -> (np.ndarray, np.ndarray):
        """Data for normalizing the data within each element."""
        means = np.zeros((self.nelems1, self.nelems2))
        stds = np.zeros((self.nelems1, self.nelems2))
        for i in range(self.nelems1):
            for j in range(self.nelems2):
                ind = self.grid.element_gll_indices(i, j)
                means[i, j] = np.mean(data[ind[0] : ind[1], ind[2] : ind[3]])
                stdi = np.std(data[ind[0] : ind[1], ind[2] : ind[3]])
                # avoid division by 0 in normalization
                stds[i, j] = stdi if stdi != 0 else 1.0
        return means, stds

    def interpolate(self, data: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Interpolate to new data points given data on the grid.

        Parameters
        ----------
            data: 2d nd.array of size self.gll1 x self.gll2
                The data at the points of the given grid, for several
                fields.

            points: 2d nd.array
                The points where we want the interpolated values.

        """
        if data.size != self.grid.gll1.size * self.grid.gll2.size:
            raise ValueError("Data is of incorrect size!")

        if points.ndim == 1:
            points = points[np.newaxis, :]

        if (
            np.max(points[:, 0]) > np.max(self.gll1)
            or np.min(points[:, 0]) < np.min(self.gll1)
            or np.max(points[:, 1]) > np.max(self.gll2)
            or np.min(points[:, 1]) < np.min(self.gll2)
        ):
            raise ValueError("Interpolation point out of bound of the mesh.")

        # Searches for element index where the points will lie
        element_ind1 = (
            np.searchsorted(self.edges1, points[:, 0], side="left") - 1
        )
        element_ind2 = (
            np.searchsorted(self.edges2, points[:, 1], side="left") - 1
        )

        # This is in case a point is said to be found to the left of the
        # left bound. This can happen when the point and the bound
        # coincide.
        # The big if above checks for actual violation of the bounds.
        # Hopefully, this combo works :-)
        for i, eli in enumerate(element_ind1):
            if eli < 0:
                element_ind1[i] = 0
        for i, eli in enumerate(element_ind2):
            if eli < 0:
                element_ind2[i] = 0

        data_means, data_stds = self.data_element_stats(data)
        values = np.zeros(points.shape[0])

        for i in range(points.shape[0]):
            eli = element_ind1[i]
            elj = element_ind2[i]
            x = points[i, 0]
            y = points[i, 1]
            edges = self.grid.element_edges(eli, elj)

            # transform point to reference element coordinates
            ref_x = (x - edges[0]) / (edges[1] - edges[0]) * 2 - 1
            ref_y = (y - edges[2]) / (edges[3] - edges[2]) * 2 - 1
            ind = self.grid.element_gll_indices(eli, elj)

            # grab data for the element
            datai = (
                data[ind[0] : ind[1], ind[2] : ind[3]] - data_means[eli, elj]
            ) / data_stds[eli, elj]

            f = np.zeros(self.lx)
            for glli in range(self.lx):
                self.interpolator.set_yi(datai[glli, :])
                f[glli] = self.interpolator(ref_y)

            self.interpolator.set_yi(f)
            values[i] = (
                self.interpolator(ref_x) * data_stds[eli, elj]
                + data_means[eli, elj]
            )

        return values
