import numpy as np
from scipy.special import roots_jacobi

__all__ = ["gll"]


def gll(n):
    """Compute the points and weights of the GLL quadrature.

    Parameters
    ----------
        n : int
            The number of points.

    """
    # Special case
    if n == 2:
        return np.array([-1, 1]), np.array([1, 1])

    x, w = roots_jacobi(n - 2, 1, 1)
    for i in range(x.size):
        w[i] /= 1 - x[i] ** 2

    x = np.append(-1, np.append(x, 1))
    w = np.append(2 / (n * (n - 1)), np.append(w, 2 / (n * (n - 1))))
    return x, w
