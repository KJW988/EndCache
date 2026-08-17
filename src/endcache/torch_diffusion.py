"""
Endpoint-hold adapter for the epsilon-prediction head in Diffusion Policy.
"""

from __future__ import annotations

import os
from typing import Literal

from endcache.core import HoldCache
from endcache.runtime import interval_from_env


CacheSpace = Literal["endpoint", "noise"]


class DDPMOutputHold:
    """
    Hold an endpoint or noise value while preserving the full DDPM grid.
    """

    def __init__(self, scheduler, reuse_interval: int, space: CacheSpace = "endpoint"):
        if space not in ("endpoint", "noise"):
            raise ValueError(f"unsupported cache space: {space!r}")
        prediction_type = getattr(scheduler.config, "prediction_type", "epsilon")
        if prediction_type != "epsilon":
            raise ValueError("DDPMOutputHold requires prediction_type='epsilon'")
        self.scheduler = scheduler
        self.space = space
        self.cache = HoldCache(reuse_interval)
        self.clip_sample = bool(getattr(scheduler.config, "clip_sample", False))

    @classmethod
    def from_environment(cls, scheduler) -> "DDPMOutputHold":
        space = os.environ.get("DP_ENDCACHE_SPACE", "endpoint").lower()
        return cls(scheduler, interval_from_env(), space=space)

    def should_refresh(self, step_index: int) -> bool:
        return self.cache.should_refresh(step_index)

    def _coefficients(self, timestep, reference):
        alpha_bar = self.scheduler.alphas_cumprod[timestep]
        alpha_bar = alpha_bar.to(device=reference.device, dtype=reference.dtype)
        alpha = alpha_bar.clamp(min=1e-8).sqrt()
        sigma = (1 - alpha_bar).clamp(min=1e-8).sqrt()
        return alpha, sigma

    def record_fresh(self, timestep, epsilon, current_state):
        """
        Convert fresh epsilon to the cache space, store it, and return epsilon.
        """
        if self.space == "noise":
            cached = epsilon
        else:
            alpha, sigma = self._coefficients(timestep, current_state)
            cached = (current_state - sigma * epsilon) / alpha
            if self.clip_sample:
                cached = cached.clamp(-1, 1)
        self.cache.store(cached.detach())
        return epsilon

    def reuse(self, timestep, current_state):
        """
        Return held noise, or reconstruct epsilon from the held endpoint and state.
        """
        cached = self.cache.value
        if self.space == "noise":
            return cached
        alpha, sigma = self._coefficients(timestep, current_state)
        return (current_state - alpha * cached) / sigma
