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
