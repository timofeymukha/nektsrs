from nektsrs.io import TimeSeries
import os
import sys


def test_ts_read():
    f = os.path.join(
        os.path.dirname(sys.modules[__name__].__file__),
        "..",
        "data",
        "test.f00001",
    )
    ts = TimeSeries.read(f)

    assert ts.npoints == 10
    assert ts.nt == 40


def test_ts_collate():
    f = os.path.join(
        os.path.dirname(sys.modules[__name__].__file__),
        "..",
        "data",
        "test.f00001",
    )
    ts = TimeSeries.read(f)
    ts.collate_data()

    assert ts.data.shape == (ts.nt, ts.nfields, ts.npoints)
