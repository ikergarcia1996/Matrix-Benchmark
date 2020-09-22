try:
    import cupy as cp

    mempool = cp.cuda.MemoryPool()
    cp.cuda.set_allocator(mempool.malloc)
    pinned_mempool = cp.cuda.PinnedMemoryPool()
    cp.cuda.set_pinned_memory_allocator(pinned_mempool.malloc)

    _cupy_available = True
    print("Cupy found, GPU available")


except ImportError:
    _cupy_available = False

    print(
        "[WARNING] Cupy not available, " "we will use CPU to perform matrix operations"
    )

import math
import numpy as np
from typing import List, Sized, Iterable, Callable, Union, Tuple
from tqdm import tqdm
import os, platform, subprocess, re


def get_gpu_memory() -> int:
    assert _cupy_available, "Cupy not available, unable to use GPU"
    _output_to_list: Callable = lambda x: x.decode("ascii").split("\n")[:-1]
    command: str = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info: List[str] = _output_to_list(
        subprocess.check_output(command.split())
    )[1:]
    memory_free_values: List[int] = [
        int(x.split()[0]) for i, x in enumerate(memory_free_info)
    ]
    return int(memory_free_values[0] * 1.049e6)  # MiB to bytes


def get_device_name(force_cpu=False):
    return get_gpu_name() if (_cupy_available and not force_cpu) else get_cpu_name()


def get_gpu_name() -> str:
    assert _cupy_available, "Cupy not available, unable to use GPU"
    _output_to_list: Callable = lambda x: x.decode("ascii").split("\n")[:-1]
    command: str = "nvidia-smi --query-gpu=gpu_name --format=csv"
    name_info: List[str] = _output_to_list(subprocess.check_output(command.split()))[1:]
    return name_info[-1]


def get_cpu_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ["PATH"] = os.environ["PATH"] + os.pathsep + "/usr/sbin"
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.decode("utf8").split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
    return ""


def get_batch_size(samples: List[np.ndarray], mem_percentage: float = 0.8) -> int:
    # Dynamic batching
    assert _cupy_available, "Cupy not available, unable to use GPU"
    assert len(samples) > 0

    samples_size: int = 0  # bytes
    for sample in samples:
        initial_bytes: int = mempool.used_bytes()
        a: cp.ndarray = cp.array(sample)
        final_bytes: int = mempool.used_bytes()
        samples_size += final_bytes - initial_bytes
        del a

    total_memory: int = get_gpu_memory() - mempool.total_bytes()
    batch_size = math.floor((total_memory / samples_size) * mem_percentage)
    print(
        f"[Dynamic Batching] Samples_size {samples_size} bytes. "
        f"Available GPU memory {total_memory} bytes. "
        f"Batch size: {batch_size}."
    )

    assert batch_size > 0, "Error not enough GPU memory for this operation"

    return batch_size


def batch(iterable: Sized, n: int = 1) -> Iterable:
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def dot(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not _cupy_available:
        return a.dot(b.T)
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], b.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            for b_index in range(0, b.shape[0], batch_size):
                a_end = min(a_index + batch_size, a.shape[0])
                b_end = min(b_index + batch_size, b.shape[0])
                result[a_index:a_end, b_index:b_end] = cp.asnumpy(
                    cp.asarray(a[a_index:a_end]).dot(cp.asarray(b[b_index:b_end]).T)
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def squared_distance(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not _cupy_available:
        return np.sum((b[np.newaxis, :] - a[:, np.newaxis]) ** 2, -1)
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], b.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Squared distance")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            for b_index in range(0, b.shape[0], batch_size):
                a_end = min(a_index + batch_size, a.shape[0])
                b_end = min(b_index + batch_size, b.shape[0])
                result[a_index:a_end, b_index:b_end] = cp.asnumpy(
                    cp.sum(
                        (
                            cp.asarray(b[b_index:b_end][np.newaxis, :])
                            - cp.asarray(a[a_index:a_end][:, np.newaxis])
                        )
                        ** 2,
                        axis=-1,
                    )
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> np.ndarray:

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    if force_cpu or not _cupy_available:
        return np.sqrt(np.sum((b[np.newaxis, :] - a[:, np.newaxis]) ** 2, -1))
    else:

        result: np.ndarray = np.zeros(
            (a.shape[0], b.shape[0]), dtype=np.result_type(a.dtype, b.dtype)
        )

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Euclidean distance")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):
            for b_index in range(0, b.shape[0], batch_size):
                a_end = min(a_index + batch_size, a.shape[0])
                b_end = min(b_index + batch_size, b.shape[0])
                result[a_index:a_end, b_index:b_end] = cp.asnumpy(
                    cp.sqrt(
                        cp.sum(
                            (
                                cp.asarray(b[b_index:b_end][np.newaxis, :])
                                - cp.asarray(a[a_index:a_end][:, np.newaxis])
                            )
                            ** 2,
                            axis=-1,
                        )
                    )
                )

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

        return result


def knn_dot(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 1,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
    ordered: bool = False,
    return_distances=False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # If ordered == True the result will be ordered by the dot product (O(n)),
    # else we will return the k-nn in a random order (O(nlog(n))

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    assert batch_size > k, "Batch size should be larger than k"

    result: np.ndarray = np.zeros((a.shape[0], k), dtype=np.int32)

    if return_distances:
        result_distances: np.ndarray = np.zeros(
            (a.shape[0], k), dtype=np.result_type(a.dtype, b.dtype)
        )
    if force_cpu or not _cupy_available:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: np.ndarray = np.zeros(
                (a_end - a_index, k),
                dtype=np.int32,
            )
            batch_result_distances: np.ndarray = np.full(
                (a_end - a_index, k),
                fill_value=np.float("-inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = a[a_index:a_end].dot(b[b_index:b_end].T)

                batch_result_indexes2 = np.argpartition(distances, kth=-k, axis=1)[
                    :, -k:
                ]
                batch_result_distances2 = distances[
                    np.arange(distances.shape[0])[:, np.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = np.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = np.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = np.argpartition(
                    new_batch_result_distances, kth=-k, axis=1
                )[:, -k:]
                batch_result_indexes = new_batch_result_indexes[
                    np.arange(new_batch_result_indexes.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    np.arange(new_batch_result_distances.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = batch_result_indexes
                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances

            else:
                indexes = np.flip(
                    batch_result_distances.argsort(axis=1),
                    axis=1,
                )

                result[a_index:a_end] = batch_result_indexes[
                    np.arange(batch_result_indexes.shape[0])[:, None], indexes
                ]

                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances[
                        np.arange(batch_result_distances.shape[0])[:, None], indexes
                    ]

    else:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: cp.ndarray = cp.zeros(
                (a_end - a_index, k),
                dtype=cp.int32,
            )
            batch_result_distances: cp.ndarray = cp.full(
                (a_end - a_index, k),
                fill_value=np.float("-inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = cp.asarray(a[a_index:a_end]).dot(
                    cp.asarray(b[b_index:b_end]).T
                )

                batch_result_indexes2 = cp.argpartition(distances, kth=-k, axis=1)[
                    :, -k:
                ]
                batch_result_distances2 = distances[
                    cp.arange(distances.shape[0])[:, cp.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = cp.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = cp.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = cp.argpartition(
                    new_batch_result_distances, kth=-k, axis=1
                )[:, -k:]
                batch_result_indexes = new_batch_result_indexes[
                    cp.arange(new_batch_result_indexes.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    cp.arange(new_batch_result_distances.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = cp.asnumpy(batch_result_indexes)
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(batch_result_distances)

            else:
                indexes = cp.flip(
                    batch_result_distances.argsort(axis=1),
                    axis=1,
                )

                result[a_index:a_end] = cp.asnumpy(
                    batch_result_indexes[
                        np.arange(batch_result_indexes.shape[0])[:, None], indexes
                    ]
                )
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(
                        batch_result_distances[
                            np.arange(batch_result_distances.shape[0])[:, None], indexes
                        ]
                    )

        del batch_result_indexes
        del batch_result_distances

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if return_distances:
        return result, result_distances
    else:
        return result


def knn_euclidean_distance(
    a: np.ndarray,
    b: np.ndarray,
    k: int = 1,
    batch_size: int = 10000,
    force_cpu: bool = False,
    show_progress: bool = False,
    ordered: bool = False,
    return_distances=False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    # If ordered == True the result will be ordered by the dot product (O(n)),
    # else we will return the k-nn in a random order (O(nlog(n))

    assert (
        a.shape[1] == b.shape[1]
    ), f"Both matrices should have the number of dimensions. A {a.shape}. B {b.shape}"

    assert batch_size > k, "Batch size should be larger than k"

    result: np.ndarray = np.zeros((a.shape[0], k), dtype=np.int32)

    if return_distances:
        result_distances: np.ndarray = np.zeros(
            (a.shape[0], k), dtype=np.result_type(a.dtype, b.dtype)
        )
    if force_cpu or not _cupy_available:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Euclidean distance KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: np.ndarray = np.zeros(
                (a_end - a_index, k),
                dtype=np.int32,
            )
            batch_result_distances: np.ndarray = np.full(
                (a_end - a_index, k),
                fill_value=np.float("inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = np.sum(
                    (b[b_index:b_end][np.newaxis, :] - a[a_index:a_end][:, np.newaxis])
                    ** 2,
                    -1,
                )

                batch_result_indexes2 = np.argpartition(distances, kth=k, axis=1)[:, :k]
                batch_result_distances2 = distances[
                    np.arange(distances.shape[0])[:, np.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = np.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = np.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = np.argpartition(
                    new_batch_result_distances, kth=k, axis=1
                )[:, :k]
                batch_result_indexes = new_batch_result_indexes[
                    np.arange(new_batch_result_indexes.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    np.arange(new_batch_result_distances.shape[0])[:, np.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = batch_result_indexes
                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances

            else:
                indexes = batch_result_distances.argsort(axis=1)

                result[a_index:a_end] = batch_result_indexes[
                    np.arange(batch_result_indexes.shape[0])[:, None], indexes
                ]

                if return_distances:
                    result_distances[a_index:a_end] = batch_result_distances[
                        np.arange(batch_result_distances.shape[0])[:, None], indexes
                    ]

    else:

        for a_index in (
            tqdm(range(0, a.shape[0], batch_size), desc="Dot product KNN")
            if show_progress
            else range(0, a.shape[0], batch_size)
        ):

            a_end = min(a_index + batch_size, a.shape[0])

            batch_result_indexes: cp.ndarray = cp.zeros(
                (a_end - a_index, k),
                dtype=cp.int32,
            )
            batch_result_distances: cp.ndarray = cp.full(
                (a_end - a_index, k),
                fill_value=np.float("inf"),
                dtype=np.result_type(a.dtype, b.dtype),
            )

            for b_index in range(0, b.shape[0], batch_size):
                b_end = min(b_index + batch_size, b.shape[0])
                distances = cp.sum(
                    (
                        cp.asarray(b[b_index:b_end][np.newaxis, :])
                        - cp.asarray(a[a_index:a_end][:, np.newaxis])
                        - cp.asarray(a[a_index:a_end][:, np.newaxis])
                    )
                    ** 2,
                    -1,
                )

                batch_result_indexes2 = cp.argpartition(distances, kth=k, axis=1)[:, :k]
                batch_result_distances2 = distances[
                    cp.arange(distances.shape[0])[:, cp.newaxis], batch_result_indexes2
                ]

                new_batch_result_distances = cp.concatenate(
                    (batch_result_distances, batch_result_distances2), axis=1
                )
                new_batch_result_indexes = cp.concatenate(
                    (batch_result_indexes, batch_result_indexes2 + b_index), axis=1
                )
                new_indexes = cp.argpartition(
                    new_batch_result_distances, kth=k, axis=1
                )[:, :k]
                batch_result_indexes = new_batch_result_indexes[
                    cp.arange(new_batch_result_indexes.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

                batch_result_distances = new_batch_result_distances[
                    cp.arange(new_batch_result_distances.shape[0])[:, cp.newaxis],
                    new_indexes,
                ]

            if not ordered:
                result[a_index:a_end] = cp.asnumpy(batch_result_indexes)
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(batch_result_distances)

            else:
                indexes = batch_result_distances.argsort(axis=1)

                result[a_index:a_end] = cp.asnumpy(
                    batch_result_indexes[
                        np.arange(batch_result_indexes.shape[0])[:, None], indexes
                    ]
                )
                if return_distances:
                    result_distances[a_index:a_end] = cp.asnumpy(
                        batch_result_distances[
                            np.arange(batch_result_distances.shape[0])[:, None], indexes
                        ]
                    )

        del batch_result_indexes
        del batch_result_distances

        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    if return_distances:
        return result, result_distances
    else:
        return result
