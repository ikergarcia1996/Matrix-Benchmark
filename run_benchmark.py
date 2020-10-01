import matrix_math as mm
import numpy as np
import time
import argparse
from typing import List


def test_distance(
    func: callable,
    matrix_size: int,
    batch_size: int,
    fp: int = 32,
    num_tests: int = 5,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> float:

    test_results: np.ndarray = np.zeros(num_tests, dtype=np.float64)

    array_type: type
    if fp == 64:
        array_type = np.float64
    elif fp == 32:
        array_type = np.float32
    elif fp == 16:
        array_type = np.float16
    else:
        raise ValueError(f"fp{fp} not supported. Available values: [16,32,64]")

    for test_no in range(num_tests):
        a: np.ndarray = np.array(np.random.rand(matrix_size, 300), dtype=array_type)
        b: np.ndarray = np.array(np.random.rand(matrix_size, 300), dtype=array_type)

        time_start: float = time.time()
        func(
            a=a,
            b=b,
            batch_size=batch_size,
            force_cpu=force_cpu,
            show_progress=show_progress,
        )
        time_end: float = time.time()

        test_results[test_no] = time_end - time_start

        del a
        del b

    return np.average(test_results)


def test_knn(
    func: callable,
    matrix_size: int,
    batch_size: int,
    k: int = 100,
    fp: int = 32,
    num_tests: int = 5,
    force_cpu: bool = False,
    show_progress: bool = False,
) -> float:

    test_results: np.ndarray = np.zeros(num_tests, dtype=np.float64)

    array_type: type
    if fp == 64:
        array_type = np.float64
    elif fp == 32:
        array_type = np.float32
    elif fp == 16:
        array_type = np.float16
    else:
        raise ValueError(f"fp{fp} not supported. Available values: [16,32,64]")

    for test_no in range(num_tests):
        a: np.ndarray = np.array(
            np.random.rand(matrix_size, matrix_size), dtype=array_type
        )
        b: np.ndarray = np.array(
            np.random.rand(matrix_size, matrix_size), dtype=array_type
        )

        time_start: float = time.time()
        func(
            a=a,
            b=b,
            k=k,
            batch_size=batch_size,
            force_cpu=force_cpu,
            show_progress=show_progress,
            ordered=True,
            return_distances=False,
        )
        time_end: float = time.time()

        test_results[test_no] = time_end - time_start

        del a
        del b

    return np.average(test_results)


def run_benchmark(
    matrix_size: int,
    batch_sizes: List[int],
    fp: int = 32,
    k: int = 100,
    num_tests: int = 5,
    force_cpu: bool = False,
    show_progress: bool = False,
    all_tasks: bool = True,
):

    print(f"---> Running benchmark <---")

    print(
        f"Device: {mm.get_device_name(force_cpu=force_cpu)}. FP{fp}. Matrix size: {matrix_size} x 300\n"
    )

    tasks_full: List[str] = [
        "dot",
        "squared_distance",
        "euclidean_distance",
        "knn_dot",
        "knn_euclidean_distance",
    ]

    tasks_minimal: List[str] = [
        "dot",
        "knn_dot",
    ]

    for task_name in tasks_full if all_tasks else tasks_minimal:
        task: callable
        task_name: str
        task_type: str

        if task_name == "dot":
            task = mm.dot
            task_type = "distance"
        elif task_name == "squared_distance":
            task = mm.squared_distance
            task_type = "distance"
        elif task_name == "euclidean_distance":
            task = mm.euclidean_distance
            task_type = "distance"
        elif task_name == "knn_dot":
            task = mm.knn_dot
            task_type = "knn"
        elif task_name == "knn_euclidean_distance":
            task = mm.knn_euclidean_distance
            task_type = "knn"
        else:
            raise ValueError(f"Task {task_name} not supported")

        for batch_size in batch_sizes:
            print(
                f"Running {task_name} task. Batch size: {batch_size}. fp{fp}. Time: ",
                end="",
            )

            try:
                if task_type == "distance":
                    result: float = test_distance(
                        func=task,
                        matrix_size=matrix_size,
                        batch_size=batch_size,
                        fp=fp,
                        num_tests=num_tests,
                        force_cpu=force_cpu,
                        show_progress=show_progress,
                    )

                else:
                    result: float = test_knn(
                        func=task,
                        matrix_size=matrix_size,
                        batch_size=batch_size,
                        k=k,
                        fp=fp,
                        num_tests=num_tests,
                        force_cpu=force_cpu,
                        show_progress=show_progress,
                    )

                print(f"{result} seconds.")

            except MemoryError:
                print("OUT OF MEMORY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu_batch_sizes",
        nargs="+",
        type=int,
        default=[100, 500, 1000, 2000, 5000, 10000],
        help="List of batch sizes to use (GPU ONLY)",
    )

    parser.add_argument(
        "--matrix_size",
        type=int,
        default=100000,
        help="Size of the matrices for the benchmarks (matrix_size x 300)",
    )
    parser.add_argument(
        "--fp",
        type=int,
        default=32,
        choices=[16, 32, 64],
        help="Floating point precision to use",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="number of nearest neighbors to retrieve in the knn benchmarks",
    )

    parser.add_argument(
        "--num_tests",
        type=int,
        default=5,
        help="Number of runs to perform for each benchmark. "
        "The final result will be the average of the score for each run",
    )

    parser.add_argument(
        "--use_cpu",
        action="store_true",
        help="Use the CPU instead of the GPU. (If this flag is not set, we will use the GPU automatically if cupy"
        "is able to find one.",
    )

    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show a progress bar during matrix operations",
    )

    parser.add_argument(
        "--full_benchmark",
        action="store_true",
        help="Run the euclidean distance / squared distance tests",
    )

    args = parser.parse_args()

    run_benchmark(
        matrix_size=args.matrix_size,
        batch_sizes=args.gpu_batch_sizes,
        fp=args.fp,
        k=args.k,
        num_tests=args.num_tests,
        force_cpu=args.use_cpu,
        show_progress=args.show_progress,
        all_tasks=args.full_benchmark,
    )
