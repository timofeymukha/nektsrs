from nektsrs.io import FileCombiner
import os
import sys


def test_file_combiner():

    f = os.path.join(
        os.path.dirname(sys.modules[__name__].__file__),
        "..",
        "data",
    )

    fc = FileCombiner("test", f)
    fc
