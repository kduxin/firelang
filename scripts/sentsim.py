import numpy as np
import numba
from numba import cuda
import multiprocessing
import logging

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.ERROR)

BLOCKDIM_X = 512

def sentsim_as_weighted_wordsim_cuda(wordsim, weight, idseqs, device=None):
    if len(idseqs) >= 1024 * 1024:
        print("You input too many sentences.")
    n = len(idseqs)
    lens = np.array([len(idseq) for idseq in idseqs])

    maxlen = max(lens)
    idseqs_arr = np.zeros((n, maxlen), dtype=np.int64)
    for i, idseq in enumerate(idseqs):
        idseqs_arr[i][: len(idseq)] = idseq

    """ columns are handled by blockIdx.x, rows are handled by blockIdx.y and threadIdx.x """
    sim = np.zeros((n, n), dtype=np.float64)
    if device is not None:
        cuda.select_device(device)
    wordsim, weight, idseqs_arr, sim, lens = list(
        map(cuda.to_device, [wordsim, weight, idseqs_arr, sim, lens])
    )

    n_block_y = (n + BLOCKDIM_X - 1) // BLOCKDIM_X
    _weightsum_kernel[(n, n_block_y), BLOCKDIM_X](
        wordsim, weight, idseqs_arr, lens, sim, n
    )

    sim = sim.copy_to_host()

    """ Fill in the upper triangle area """
    diag = np.diag(sim)
    sim = sim + sim.T
    sim[np.diag_indices(len(diag))] = diag
    return sim


@cuda.jit
def _weightsum_kernel(wordsim, weight, idseqs, lens, out, n):
    x = cuda.blockIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.x + cuda.threadIdx.x
    if x >= n or y >= n:
        return
    """ Compute the lower triangle area (including the diagonal) only """
    if x > y:
        return

    len1 = lens[x]
    weight1sum = 0.0
    for i in range(len1):
        weight1sum += weight[idseqs[x, i]]

    len2 = lens[y]
    weight2sum = 0.0
    for j in range(len2):
        weight2sum += weight[idseqs[y, j]]

    sim: float = 0
    for i in range(len1):
        for j in range(len2):
            sim += (
                (weight[idseqs[x, i]] / weight1sum)
                * (weight[idseqs[y, j]] / weight2sum)
                * wordsim[idseqs[x, i], idseqs[y, j]]
            )
    out[x, y] = sim


def sentsim_as_weighted_wordsim_cpu(wordsim, weight, idseqs):
    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    idseqs = [np.array(idseq) for idseq in idseqs]

    n = len(idseqs)
    num_workers = max(8, min(32, n // 100 + 1))
    with multiprocessing.Pool(
        num_workers, initializer=_init, initargs=(wordsim, weight, idseqs)
    ) as pool:
        tasks = [(i, j) for i in range(n) for j in range(i, n)]
        sim_batches = pool.map(overij_batch, chunker(tasks, 1000))
    sim_values = np.array([x for y in sim_batches for x in y])

    triu_indices = np.triu_indices(n)
    sim = np.zeros((n, n), dtype=np.float32)
    sim[triu_indices] = sim_values
    sim = sim.T
    sim[triu_indices] = sim.T[triu_indices]
    return sim


cache = []


def _init(wordsim, weight, idseqs):
    cache.extend([wordsim, weight, idseqs])


@numba.njit
def _sentsim(ids1, ids2, wordsim, weight):
    weight1 = weight[ids1]
    weight1 /= weight1.sum()
    weight2 = weight[ids2]
    weight2 /= weight2.sum()

    ids = ids1.reshape(-1, 1) * wordsim.shape[1] + ids2.reshape(1, -1)
    sim = wordsim.take(ids.reshape(-1))  # .reshape(len(ids1), len(ids2))
    return (sim * (weight1.reshape(-1, 1) * weight2.reshape(1, -1)).reshape(-1)).sum()
    # return weight1 @ wordsim[ids1][:, ids2] @ weight2


def overij_batch(ijs):
    wordsim, weight, idseqs = cache
    simij = np.array([_sentsim(idseqs[i], idseqs[j], wordsim, weight) for i, j in ijs])
    return simij


if __name__ == "__main__":

    wordsim = np.abs(np.random.randn(10, 10)) + 1e-5
    weight = np.abs(np.random.randn(10)) + 1e-5
    idseqs = [
        [1, 2, 3, 4],
        [5, 6, 7],
        [2, 0, 8],
    ] * 512

    res1 = sentsim_as_weighted_wordsim_cpu(wordsim, weight, idseqs)
    res2 = sentsim_as_weighted_wordsim_cuda(wordsim, weight, idseqs)
    print(res1)
    print(res2)
