import numpy as np
import struct
from typing import List
from .helpers import read_int, read_real

__all__ = ["TimeSeries"]


class TimeSeries:
    """Data from a single pts file.

    Use the class method read() to read in the data.

    The data field will be a list with data corresponding to each
    point. Use the collate_data() method to collate that to a single
    ndarray. Note: this has to be done prior to calling save()!
    """

    def __init__(
        self,
        data: List,
        t: List,
        ldim: int,
        writetime: float,
        nfields: int,
        sort: bool = True,
    ) -> None:
        self.data = data
        self.t = np.array(t)
        self.ldim = ldim
        self.writetime = writetime
        self.npoints = len(data)
        self.nfields = nfields
        self.nt = self.t.size
        self.sort = sort

        if sort:
            self.data.sort(key=lambda item: item.id)

        self.locs = np.vstack([data[i].pos for i in range(self.npoints)])
        self.id_list = np.array([data[i].id for i in range(self.npoints)])

    @classmethod
    def read(cls, fname: str, sort: bool = True):
        """Read data from an interpolation file"""
        infile = open(fname, "rb")

        header = infile.read(132).split()

        # extract word size
        wdsizet = int(header[1])
        wdsizef = int(header[2])

        # identify endian encoding
        etagb = infile.read(4)
        etag_l = struct.unpack("<f", etagb)[0]
        etag_l = int(etag_l * 1e5) / 1e5
        etag_b = struct.unpack(">f", etagb)[0]
        etag_b = int(etag_b * 1e5) / 1e5
        if etag_l == 6.54321:
            emode = "<"
        elif etag_b == 6.54321:
            emode = ">"
        else:
            raise ValueError("Could not determine endian")

        # get simulation parameters
        ldim = int(header[3])
        npoints = int(header[5])
        nt = int(header[6])
        nfields = int(header[7])
        time = float(header[8])

        # create main data structure filled with 0
        data = [Point(ldim, nt, nfields) for _ in range(npoints)]

        # read snapshot time list
        timelist = read_real(infile, emode, wdsizet, nt)

        # read global point number
        # NOTE: I convert to 0-based numbering
        global_id_list = read_int(infile, emode, npoints)
        for i in range(npoints):
            data[i].id = global_id_list[i] - 1

        # read coordinates
        for i in range(npoints):
            pos = read_real(infile, emode, wdsizet, ldim)
            data[i].pos = pos

        # read fields
        for i in range(npoints):
            for j in range(nt):
                fld = read_real(infile, emode, wdsizef, nfields)
                for k in range(nfields):
                    data[i].data[j][k] = fld[k]

        return cls(data, timelist, ldim, time, nfields, sort)

    def collate_data(self) -> None:
        """Combine data from all points in a single data array."""
        self.data = np.stack(
            [self.data[i].data for i in range(self.npoints)], axis=2
        )

    def save(self, filepath: str) -> None:
        """Save the data to an hdf5 file."""
        import h5py

        f = h5py.File(filepath, "w")
        f.create_dataset("locs", data=self.locs)
        f.create_dataset("t", data=self.t)
        f.create_dataset("data", data=self.data)

        f.attrs["writetime"] = self.writetime
        f.attrs["nfields"] = self.nfields
        f.attrs["npoints"] = self.npoints
        f.attrs["ldim"] = self.ldim
        f.attrs["nt"] = self.nt

        f.close()


class Point:
    """Data at a single point."""

    def __init__(self, ldim: int, nt: int, nfields: int):
        self.id = int(np.zeros(1, dtype=np.uint32)[0])
        self.pos = np.zeros(ldim)
        self.data = np.zeros((nt, nfields))
