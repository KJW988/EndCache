from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Callable, List


@dataclass(frozen=True)
class LatencySummary:
    mean_ms: float
    std_ms: float
    samples_ms: List[float]


def benchmark_cuda_events(
    operation: Callable[[], object], warmup: int = 30, repeats: int = 100
) -> LatencySummary:
    """
    Measure one prepared operation with CUDA events.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for CUDA-event latency measurement")
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return LatencySummary(mean(samples), pstdev(samples), samples)
