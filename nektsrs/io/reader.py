import numpy as np
import struct

__all__ = ["Reader"]

class Reader:
    def __init__(self, file: str):
        self.file = file

    @classmethod
    def read(cls, file: str):
        pass

    @classmethod
    def read_int(infile, emode, nvar):
        """read integer array"""
        isize = 4
        llist = infile.read(isize * nvar)
        llist = list(struct.unpack(emode + nvar * 'i', llist))
        return llist

    @classmethod
    def read_flt(infile, emode, wdsize, nvar):
        """read real array"""
        if (wdsize == 4):
            realtype = 'f'
        elif (wdsize == 8):
            realtype = 'd'
        llist = infile.read(wdsize * nvar)
        llist = np.frombuffer(llist, dtype=emode + realtype, count=nvar)
        return llist


    class Point:
        """Data at a single point."""

        def __init__(self, ndim: int, nt: int, nfields: int):
            self.glid = np.zeros((1), dtype=np.uint32)
            self.pos = np.zeros((ndim))
            self.fld = np.zeros((nt, nfields))


