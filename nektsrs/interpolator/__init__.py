from .interpolator1d import *
from .interpolator2d import *

__all__ = ["interpolator1d", "interpolator2d"]
__all__.extend(interpolator1d.__all__)
__all__.extend(interpolator2d.__all__)
