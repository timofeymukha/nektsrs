import numpy as np
import h5py
import glob
from os.path import join
from typing import Optional
from nektsrs.io import TimeSeries

__all__ = ["FileCombiner"]


class FileCombiner:
    def __init__(self, basename: str, basepath: Optional[str] = "") -> None:

        search_string = join(
            basepath, "pts" + basename + "[0-1].f[0-9][0-9][0-9][0-9][0-9]"
        )
        datafiles = glob.glob(search_string)

        print(f"Found {len(datafiles)} datafiles")
        if len(datafiles) == 0:
            raise FileExistsError("Could not find any data!")

        datasets = [TimeSeries.read(i) for i in datafiles]

        # collate the data to a 3d array for each dataset
        for i in datasets:
            i.collate_data()

        writetimes = [i.writetime for i in datasets]
        print("Datasets written at the following timesteps were found")
        print(writetimes)

        datasets.sort(key=lambda item: item.writetime)

        self.data = np.vstack([i.data for i in datasets])

        self.t = np.concatenate([i.t for i in datasets])
        self.timespan = np.array([self.t[0], self.t[-1]])
        self.nfields = self.data.shape[1]
        self.npoints = self.data.shape[2]
        self.ldim = datasets[0].ldim
        self.locs = datasets[0].locs

        # kill overlap values
        _, idx = np.unique(self.t, return_index=True)
        self.data = self.data[idx, ...]
        self.t = self.t[idx]
        self.nt = self.t.size

        print(f"Final shape of the data is {self.data.shape}")

    def save(self, filepath: str) -> None:
        """Save the data to an hdf5 file."""
        f = h5py.File(filepath, "w")
        f.create_dataset("locs", data=self.locs)
        f.create_dataset("t", data=self.t)
        f.create_dataset("data", data=self.data)

        f.attrs["timespan"] = self.timespan
        f.attrs["nfields"] = self.nfields
        f.attrs["npoints"] = self.npoints
        f.attrs["ldim"] = self.ldim
        f.attrs["nt"] = self.nt

        f.close()
