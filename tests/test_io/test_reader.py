from nektsrs.io import TimeSeries


def test_reader_read():
    ts = TimeSeries.read(
        "C:\\Users\\tiamm\\Nek5000\\run\\tests\\ptsvs_vrm_ref00.f00001"
    )

    assert ts.npoints == 10
    assert ts.nt == 40


def test_reader_collate():
    ts = TimeSeries.read(
        "C:\\Users\\tiamm\\Nek5000\\run\\tests\\ptsvs_vrm_ref00.f00001"
    )

    ts.collate_data()

    assert ts.data.shape == (ts.nt, ts.nfields, ts.npoints)
