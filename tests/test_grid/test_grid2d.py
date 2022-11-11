from nektsrs.grid import Grid2D, SimpleGrid2D
from nektsrs.gll import gll
from numpy.testing import assert_array_almost_equal
import numpy as np


def test_simplegrid2d():
    g = SimpleGrid2D(0, 2, 0, 1, 5, 3, 4)
    assert g.lx == 4
    assert g.n1 == 5
    assert g.n2 == 3
    assert g.start1 == 0
    assert g.end1 == 2
    assert g.start2 == 0
    assert g.end2 == 1


def test_simplegrid2d_1elem():
    g = SimpleGrid2D(-1, 1, -1, 1, 1, 1, 4)
    assert g.lx == 4
    assert g.n1 == 1
    assert g.n2 == 1
    assert g.start1 == -1
    assert g.end1 == 1
    assert g.start2 == -1
    assert g.end2 == 1

    p, _ = gll(4)
    assert_array_almost_equal(g.gll[:4, 0], p)


def test_grid2d():
    e = np.linspace(0, 1, 5)
    g = Grid2D(e, e, 4)
    assert g.lx == 4
    assert g.n1 == 4
    assert g.n2 == 4
    assert g.start1 == 0
    assert g.end1 == 1
    assert g.start2 == 0
    assert g.end2 == 1

    g2 = SimpleGrid2D(0, 1, 0, 1, 4, 4, 4)
    assert_array_almost_equal(g.gll, g2.gll)
