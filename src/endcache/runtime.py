import os


def _positive_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    value = int(default if raw is None else raw)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def interval_from_env(default: int = 1) -> int:
    return _positive_env("ENDCACHE_INTERVAL", default)


def steps_from_env(default: int) -> int:
    return _positive_env("ENDCACHE_NUM_STEPS", default)
