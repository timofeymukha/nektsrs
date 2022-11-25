import numpy as np
from mpi4py import MPI
import h5py
from tqdm import trange
from nektsrs.grid import SimpleGrid1D
from nektsrs.chunks import chunks_and_offsets
from nektsrs.interpolator import Interpolator1D
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="A utility to interpolate point data \
                             on a gll grid to a uniform grid."
    )

    parser.add_argument(
        "--input",
        type=str,
        help="The input hdf5 file with the data on the gll grid.",
        required=True,
    )

    parser.add_argument(
        "--output",
        type=str,
        help="The output hdf5 file with the interpolated data.",
        required=True,
    )

    parser.add_argument(
        "--lenx",
        type=float,
        help="The domain length in x, i.e. the first axis.",
        required=True,
    )

    parser.add_argument(
        "--lenz",
        type=float,
        help="The domain length in z, i.e. the second axis.",
        required=True,
    )

    parser.add_argument(
        "--nx",
        type=int,
        help="The # of elements in x, i.e. the first axis.",
        required=True,
    )

    parser.add_argument(
        "--nz",
        type=int,
        help="The # of elements in z, i.e. the second axis.",
        required=True,
    )

    parser.add_argument(
        "--lx",
        type=int,
        help="The # of gll points per elements.",
        required=True,
    )

    args = parser.parse_args()

    input_file = args.input
    output_file = args.output
    length_x = args.lenx
    length_z = args.lenz
    nx = args.nx
    nz = args.nz
    lx = args.lx

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # The file from which we interpolate
    pts = h5py.File(input_file, "r", driver="mpio", comm=comm)

    # Create 1D grids corresponding to what we have in the
    # simulation as per the .box file
    eps = 1e-7
    gridx = SimpleGrid1D(eps, length_x - eps, n=nx, lx=lx)
    gridz = SimpleGrid1D(eps, length_z - eps, n=nz, lx=lx)

    npx = gridx.gll.size
    npz = gridz.gll.size

    # The interpolator
    intp = Interpolator1D(gridx)

    # interpolation points
    points = np.linspace(eps, length_x - eps, npx)
    nt = pts["data"].shape[0]
    nt = 10

    f = h5py.File(output_file, "w", driver="mpio", comm=comm)

    new_data = f.create_dataset("data", (nt, npx, npz))
    f.create_dataset("t", data=pts["t"][:nt])

    if rank == 0:
        print(
            f"The time span of the data is {np.min(pts['t'])}, \
              {np.max(pts['t'])}"
        )

    [chunks, offsets] = chunks_and_offsets(nprocs, nt)

    temp = np.zeros((npx, npz))

    if rank == 0:
        loop_range = trange(chunks[rank])
    else:
        loop_range = range(chunks[rank])

    # Each rank loops through its portion of the time-index
    for i in loop_range:
        position = offsets[rank] + i

        # Loop across z index and do 1D interpolation in x
        datai = np.reshape(pts["data"][position, 0, :], (npx, npz), "F")
        for i in range(npz):
            temp[:, i] = intp.interpolate(datai[:, i], points)

        new_data[position, :, :] = temp

    comm.Barrier()
    f.close()
    pts.close()
    if rank == 0:
        print(rank, "Done")


if __name__ == "__main__":
    main()
