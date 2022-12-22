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


from typing import List
import struct
import numpy as np

__all__ = ["read_int", "read_real"]


def read_int(infile, emode: str, nvar: int) -> List[int]:
    """Read an integer array."""
    isize = 4
    llist = infile.read(isize * nvar)
    return list(struct.unpack(emode + nvar * "i", llist))


def read_real(
    infile,
    emode: str,
    wdsize: int,
    nvar: int,
) -> List[float]:
    """Read a real array."""
    if wdsize == 4:
        realtype = "f"
    elif wdsize == 8:
        realtype = "d"
    else:
        raise ValueError

    llist = infile.read(wdsize * nvar)
    return np.frombuffer(llist, dtype=emode + realtype, count=nvar)
