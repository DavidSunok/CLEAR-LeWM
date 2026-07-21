from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _as_trace(values) -> np.ndarray:
    trace = np.asarray(values, dtype=bool)
    if trace.ndim == 1:
        trace = trace[:, None]
    if trace.ndim != 2:
        raise ValueError(
            "Success values must have shape [episodes] or [episodes, steps]"
        )
    return trace


def sustained_outcomes(trace, hold_steps: int) -> np.ndarray:
    trace = _as_trace(trace)
    if hold_steps <= 0:
        raise ValueError("hold_steps must be positive")
    if hold_steps == 1:
        return trace.any(axis=1)
    if trace.shape[1] < hold_steps:
        return np.zeros(trace.shape[0], dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(
        trace.astype(np.int8), window_shape=hold_steps, axis=1
    )
    return (windows.sum(axis=-1) == hold_steps).any(axis=1)


def bootstrap_ci(
    values: np.ndarray,
    seed: int = 0,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    batch = 1_000
    for offset in range(0, samples, batch):
        count = min(batch, samples - offset)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[offset : offset + count] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def summarize_success(
    success_trace,
    random_trace=None,
    hold_steps: int = 1,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> dict:
    trace = _as_trace(success_trace)
    first_hit = trace.any(axis=1)
    final = trace[:, -1]
    sustained = sustained_outcomes(trace, hold_steps)
    low, high = bootstrap_ci(first_hit, seed=seed, samples=bootstrap_samples)
    result = {
        "episodes": int(len(trace)),
        "success_rate_percent": float(first_hit.mean() * 100.0),
        "success_rate_ci95_percent": [low * 100.0, high * 100.0],
        "final_state_success_rate_percent": float(final.mean() * 100.0),
        "sustained_success_rate_percent": float(sustained.mean() * 100.0),
        "sustained_steps": int(hold_steps),
    }

    if random_trace is not None:
        random_values = _as_trace(random_trace).any(axis=1)
        if len(random_values) != len(first_hit):
            raise ValueError("Method and random traces must use the same manifest")
        method_sr = first_hit.mean() * 100.0
        random_sr = random_values.mean() * 100.0
        denominator = 100.0 - random_sr
        paired = first_hit.astype(np.float64) - random_values.astype(np.float64)
        gain_low, gain_high = bootstrap_ci(paired, seed=seed, samples=bootstrap_samples)
        result.update(
            {
                "random_success_rate_percent": float(random_sr),
                "excess_over_random_pp": float(method_sr - random_sr),
                "normalized_success_percent": (
                    float((method_sr - random_sr) / denominator * 100.0)
                    if denominator > 0
                    else None
                ),
                "paired_gain_ci95_pp": [gain_low * 100.0, gain_high * 100.0],
            }
        )
    return result


def load_success_trace(path: str | Path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return data
    for key in ("success_trace", "episode_successes", "successes"):
        if key in data:
            return data[key]
    if "metrics" in data and "episode_successes" in data["metrics"]:
        return data["metrics"]["episode_successes"]
    raise KeyError(f"Could not find success values in {path}")
