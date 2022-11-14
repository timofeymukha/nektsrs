import numpy as np
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
        self.locs = datasets[0].locs

        # kill overlap values
        _, idx = np.unique(self.t, return_index=True)
        self.data = self.data[idx, ...]
        self.t = self.t[idx]
