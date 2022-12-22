import numpy as np
from mpi4py import MPI
import h5py
import glob
from os.path import join
from nektsrs.io import TimeSeries
from nektsrs.chunks import chunks_and_offsets
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="A utility to interpolate point data \
                             on a gll grid to a uniform grid."
    )

    parser.add_argument(
        "--basename",
        type=str,
        help="The base string, which follows pts in the file names",
        required=True,
    )

    parser.add_argument(
        "--output", type=str, help="The output hdf5 file.", required=True
    )

    args = parser.parse_args()
    basename = args.basename
    output_file = args.output

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    search_string = join("pts" + basename + "[0-1].f[0-9][0-9][0-9][0-9][0-9]")
    datafiles = glob.glob(search_string)

    if rank == 0:
        print(f"Found {len(datafiles)} datafiles")

    if len(datafiles) == 0:
        raise FileExistsError(f"Rank {rank} could not find any data!")

    [chunks, offsets] = chunks_and_offsets(nprocs, len(datafiles))

    if rank == 0:
        print("Reading...")
    datasets = []
    for i in range(chunks[rank]):
        position = offsets[rank] + i
        datasets.append(TimeSeries.read(datafiles[position]))
        datasets[i].collate_data()

    print(rank, chunks[rank])
    # print("length", rank, len(datasets))

    writetimes = [i.writetime for i in datasets]
    # print("Datasets written at the following timesteps were found")
    # print(writetimes)

    datasets.sort(key=lambda item: item.writetime)
    writetimes.sort()

    data = np.vstack([i.data for i in datasets])
    t = np.concatenate([i.t for i in datasets])

    f = h5py.File(output_file, "w", driver="mpio", comm=comm)

    times = comm.gather(t[0], root=0)
    lengths = np.array(comm.gather(t.size, root=0))

    if rank == 0:
        starts = np.zeros(nprocs, dtype=np.int64)
    else:
        starts = np.empty(nprocs, dtype=np.int64)

    if rank == 0:
        print("Computing starting indices in global array")
        starts = np.zeros(nprocs, dtype=np.int64)
        sort_ind = np.argsort(times)

        #    print("times", times)
        #    print("lengths", lengths)

        for i in range(starts.size):
            pos = np.where(sort_ind == i)[0][0]
            starts[i] = np.sum(lengths[:pos])

    if rank == 0:
        nt = np.array([np.sum(lengths)], dtype="i")
    else:
        nt = np.empty(1, dtype="i")
    comm.Bcast(nt, root=0)

    comm.Bcast(starts, root=0)

    if rank == 0:
        print("Writing data")
    f_time = f.create_dataset("t", (nt))
    f_data = f.create_dataset("data", (nt, data.shape[1], data.shape[2]))

    print(rank, starts[rank], starts[rank] + t.size, t.size, nt, data.shape)

    f_time[starts[rank] : starts[rank] + t.size] = t
    f_data[starts[rank] : starts[rank] + t.size] = data

    comm.Barrier()
    f.close()

    if rank == 0:
        print("Done")


# self.timespan = np.array([self.t[0], self.t[-1]])
# self.nfields = self.data.shape[1]
# self.npoints = self.data.shape[2]
# self.ldim = datasets[0].ldim
# self.locs = datasets[0].locs

# kill overlap values
# _, idx = np.unique(self.t, return_index=True)
# self.data = self.data[idx, ...]
# self.t = self.t[idx]
# self.nt = self.t.size


if __name__ == "__main__":
    main()
