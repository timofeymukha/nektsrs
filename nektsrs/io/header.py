__all__ = ["Header"]


class Header:
    def __init__(self, file: str) -> None:
        infile = open(file, "rb")

        header = infile.read(132).split()

        # get simulation parameters
        self.ldim = int(header[3])
        self.npoints = int(header[5])
        self.nt = int(header[6])
        self.nfields = int(header[7])
        self.time = float(header[8])
        infile.close()
