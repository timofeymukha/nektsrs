import numpy as np

__all__ = ["chunks_and_offsets"]


def chunks_and_offsets(nprocs, size):
    """Given the size of a 1d array and the number of processors,
    compute chunk-sizes for each processor and the starting indices
    (offsets) for each processor.

    Parameters
    ----------
    nprocs : int
        The amount of processors.
    size : int
        The size of the 1d array to be distributed.

    Returns
    -------
    List of two ndarrays.
        The first array contains the chunk-size for each processor.
        The second array contains the offset (starting index) for
        each processor.
    """

    # To ensure integer division later
    nprocs = int(nprocs)

    if nprocs < 0 or nprocs > size:
        raise ValueError("Number of processors is invalid", nprocs, size)

    chunks = np.zeros(nprocs, dtype=np.int64)
    nrAlloced = 0

    for i in range(nprocs):
        remainder = size - nrAlloced
        buckets = nprocs - i
        chunks[i] = remainder / buckets
        nrAlloced += chunks[i]

    # Calculate the offset for each processor
    offsets = np.zeros(chunks.shape, dtype=np.int64)

    for i in range(offsets.shape[0] - 1):
        offsets[i + 1] = np.sum(chunks[: i + 1])

    if np.sum(chunks) != size:
        raise ValueError

    return [chunks, offsets]
