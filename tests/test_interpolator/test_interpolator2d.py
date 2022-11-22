import numpy as np
from nektsrs.interpolator import Interpolator2D, Interpolator1D
from nektsrs.grid import SimpleGrid2D, SimpleGrid1D
from numpy.testing import assert_array_almost_equal


def test_interpolator2d_constant():
    g = SimpleGrid2D(-1, 1, -1, 1, 1, 1, 4)
    intp = Interpolator2D(g)

    data = np.ones((g.gll1.size, g.gll2.size))
    p = np.array([0, 1])
    v = intp.interpolate(data, p)
    assert_array_almost_equal([1], v[0])


def test_interpolator2d_linear():
    g = SimpleGrid2D(-1, 1, -1, 1, 1, 1, 4)
    g1d = SimpleGrid1D(-1, 1, 1, 4)

    intp = Interpolator2D(g)
    intp1d = Interpolator1D(g1d)

    data = np.ones((g.gll1.size, g.gll2.size))

    for i, gll1i in enumerate(g.gll1):
        for j, gll1j in enumerate(g.gll2):
            data[i, j] = gll1i + gll1j

    x = np.linspace(-1, 1, 5)

    for i, gi in enumerate(g.gll2):
        p = np.stack((x, np.full(x.size, gi)), axis=1)
        v = intp.interpolate(data, p)
        v1 = intp1d.interpolate(data[:, i], x)
        assert_array_almost_equal(v, v1)
    #    assert_array_almost_equal(v, data)

    pass
