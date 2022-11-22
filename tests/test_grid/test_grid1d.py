from nektsrs.grid import SimpleGrid1D, Grid1D
from nektsrs.gll import gll
from numpy.testing import assert_array_almost_equal
import numpy as np


def test_simplegrid1d_element_edges():
    g = SimpleGrid1D(0, 1, 5, 4)

    assert_array_almost_equal(g.element_edges(0), (0, 0.2))
    assert_array_almost_equal(g.element_edges(2), (0.4, 0.6))


def test_simplegrid1d_element_gll():
    g = SimpleGrid1D(0, 1, 5, 4)
    for i in range(g.n - 1):
        gll = g.element_gll_points(i)
        assert gll[0] == g.edges[i]
        assert gll[-1] == g.edges[i + 1]


def test_simplegrid1d():
    g = SimpleGrid1D(0, 1, 10, 8)
    assert g.lx == 8
    assert g.n == 10
    assert g.start == 0
    assert g.end == 1


def test_simplegrid1d_1elem():
    g = SimpleGrid1D(-1, 1, 1, 8)

    assert g.lx == 8
    assert g.n == 1
    assert g.start == -1
    assert g.end == 1

    p, _ = gll(8)
    assert_array_almost_equal(g.gll, p)


def test_grid1d():
    e = np.linspace(0, 1, 11)
    g = Grid1D(e, 4)

    assert g.lx == 4
    assert g.n == 10
    assert g.start == 0
    assert g.end == 1

    p, _ = gll(4)
    assert_array_almost_equal(g.gll[:4], (p + 1) / 2 * 0.1)
