import numpy as np
import struct

__all__ = ["Writer"]


class Writer:
    def __init__(self, locs: np.ndarray):
        self.locs = locs

    def write(self, fname: str = "int_pos", wdsize: int = 8, emode: str = "<"):
        npoints = self.locs.shape[0]

        outfile = open(fname, "wb")

        if wdsize == 4:
            realtype = "f"
        elif wdsize == 8:
            realtype = "d"
        else:
            raise ValueError

        # header
        header = "#iv1 %1i %1i %10i " % (wdsize, self.locs.shape[1], npoints)
        header = header.ljust(32)
        outfile.write(header.encode("utf-8"))

        # write tag (to specify endianness)
        outfile.write(struct.pack(emode + "f", 6.54321))

        # write point positions
        for i in range(npoints):
            pointi = self.locs[i, :]
            outfile.write(
                struct.pack(emode + self.locs.shape[1] * realtype, *pointi)
            )
