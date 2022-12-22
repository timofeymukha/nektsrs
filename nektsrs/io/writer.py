#    Copyright 2022 eX-Mech/pymech developers, Timofey Mukha
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
