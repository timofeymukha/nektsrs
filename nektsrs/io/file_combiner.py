import numpy as np
import glob
from os.path import join
from typing import Optional
from nektsrs.io import TimeSeries

__all__ = ["FileCombiner"]


class FileCombiner:
    def __init__(self, basename: str, basepath: Optional[str] = "") -> None:

        datafiles = glob.glob(
            join(
                basepath, "pts" + basename + "[0-1].f[0-9][0-9][0-9][0-9][0-9]"
            )
        )

        print(f"Found {len(datafiles)} datafiles")

        datasets = [TimeSeries.read(i) for i in datafiles]

        # collate the data to a 3d array for each dataset
        for i in datasets:
            i.collate_data()

        writetimes = [i.writetime for i in datasets]
        print("Datasets written at the following timesteps were found")
        print(writetimes)

        datasets.sort(key=lambda item: item.writetime)

        self.data = (np.row_stack([i.data for i in datasets])[0],)
        print(type(self.data), type(datasets[0].data), len(self.data))
        print(self.data)

        self.t = (np.stack([i.t for i in datasets]),)
        self.timespan = np.array([self.t[0], self.t[-1]])
        self.nfields = self.data.shape[1]
        self.locs = datasets[0].locs

        # kill overlap values
        idx = np.unique(self.t, return_index=True)
        print(idx)
        self.data = self.data[idx, ...]
        self.t = self.t[idx]
