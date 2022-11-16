import numpy as np

from nektsrs.interpolator import Interpolator1D
from nektsrs.grid import SimpleGrid1D
from numpy.testing import assert_array_almost_equal


def test_interpolator1d_element_edges():
    g = SimpleGrid1D(0, 1, 5, 4)
    intp = Interpolator1D(g)

    assert_array_almost_equal(intp.element_edges(0), (0, 0.2))
    assert_array_almost_equal(intp.element_edges(2), (0.4, 0.6))


def test_interpolator1d_element_gll():
    g = SimpleGrid1D(0, 1, 5, 4)
    intp = Interpolator1D(g)

    for i in range(intp.nelems - 1):
        gll = intp.element_gll_points(i)
        assert gll[0] == intp.edges[i]
        assert gll[-1] == intp.edges[i + 1]


def test_interpolator1d():

    g = SimpleGrid1D(0, 1, 5, 4)
    intp = Interpolator1D(g)

    data = np.ones(g.gll.size)
    v = intp.interpolate(data, 0.1)
    assert_array_almost_equal([1], v[0])

    # linear function
    data = g.gll
    p = np.linspace(0, 0.1, 5)
    v = intp.interpolate(data, p)
    assert_array_almost_equal(v, p)

    g = SimpleGrid1D(0, 12, 32, 8)
    data = np.random.randn(g.gll.size) * np.cos(g.gll)
    intp = Interpolator1D(g)
    v = intp.interpolate(data, g.gll)
    assert_array_almost_equal(v, data)

    pass
