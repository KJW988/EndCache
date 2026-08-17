from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, Tuple, TypeVar, cast


T = TypeVar("T")
S = TypeVar("S")


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def nfe_count(num_steps: int, reuse_interval: int) -> int:
    """
    Return the function-evaluation count ``ceil(N / k)`` for N solver steps.
    """
    num_steps = _positive_int(num_steps, "num_steps")
    reuse_interval = _positive_int(reuse_interval, "reuse_interval")
    return (num_steps + reuse_interval - 1) // reuse_interval


def refresh_indices(num_steps: int, reuse_interval: int) -> Tuple[int, ...]:
    """
    Return the zero-based refresh-step indices.
    """
    num_steps = _positive_int(num_steps, "num_steps")
    reuse_interval = _positive_int(reuse_interval, "reuse_interval")
    return tuple(range(0, num_steps, reuse_interval))


@dataclass
class HoldCache(Generic[T]):
    """
    Zero-order hold cache that stores the most recent fresh output.
    """

    reuse_interval: int
    _value: Optional[T] = None

    def __post_init__(self) -> None:
        self.reuse_interval = _positive_int(self.reuse_interval, "reuse_interval")

    def reset(self) -> None:
        self._value = None

    @property
    def initialized(self) -> bool:
        return self._value is not None

    def should_refresh(self, step_index: int) -> bool:
        if step_index < 0:
            raise ValueError(f"step_index must be >= 0, got {step_index}")
        return not self.initialized or step_index % self.reuse_interval == 0

    def store(self, value: T) -> T:
        self._value = value
        return value

    @property
    def value(self) -> T:
        if not self.initialized:
            raise RuntimeError("cache has no fresh value")
        return cast(T, self._value)

    def resolve(self, step_index: int, evaluate: Callable[[], T]) -> T:
        if self.should_refresh(step_index):
            self.store(evaluate())
        return self.value


def run_hold_solver(
    initial_state: S,
    num_steps: int,
    reuse_interval: int,
    evaluate: Callable[[int, S], T],
    solver_step: Callable[[int, S, T], S],
) -> S:
    num_steps = _positive_int(num_steps, "num_steps")
    cache: HoldCache[T] = HoldCache(reuse_interval)
    state = initial_state
    for step_index in range(num_steps):
        output = cache.resolve(step_index, lambda: evaluate(step_index, state))
        state = solver_step(step_index, state, output)
    return state
