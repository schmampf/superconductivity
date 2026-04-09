from __future__ import annotations

from pathlib import Path
import threading

import numpy as np

from superconductivity.models.mar.core.cache import ensure_curve_cached
from superconductivity.models.mar.core.cache import explore_curve_store
from superconductivity.models.mar.core.cache import load_curve
from superconductivity.models.mar.core.cache import save_curve


def test_mar_cache_serializes_threaded_access(tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.h5"
    barrier = threading.Barrier(4)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        try:
            barrier.wait()
            group_path = f"ha_sym/curves/thread-{index}"
            attrs = {
                "model": "ha_sym",
                "tau": float(index),
                "T_K": 0.05,
                "Delta_meV": 0.18,
                "gamma_meV": 1e-4,
            }
            for repeat in range(5):
                V_q = np.array([index * 100 + repeat + 1], dtype=np.int64)
                I_nA = np.array([float(index + repeat)], dtype=np.float64)
                save_curve(
                    cache_file=cache_file,
                    group_path=group_path,
                    attrs=attrs,
                    V_q=V_q,
                    I_nA=I_nA,
                )
                V_loaded_q, I_loaded_nA = load_curve(
                    cache_file=cache_file,
                    group_path=group_path,
                )
                np.testing.assert_array_equal(V_loaded_q, V_q)
                np.testing.assert_allclose(I_loaded_nA, I_nA)
        except BaseException as exc:  # pragma: no cover - exercised on failure.
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(index,))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if errors:
        raise errors[0]

    assert cache_file.with_name("cache.h5.lock").exists()
    summary = explore_curve_store(
        cache_file=cache_file,
        root_group_path="ha_sym/curves",
    )
    assert len(summary) == 4


def test_ensure_curve_cached_only_evaluates_missing_bins(
    tmp_path: Path,
) -> None:
    cache_file = tmp_path / "cache.h5"
    attrs = {
        "model": "ha_sym",
        "tau": 0.4,
        "T_K": 0.05,
        "Delta_meV": 0.18,
        "gamma_meV": 1e-4,
    }
    calls: list[np.ndarray] = []

    def evaluate_missing_q(V_missing_q: np.ndarray) -> np.ndarray:
        calls.append(V_missing_q.copy())
        return V_missing_q.astype(np.float64)

    V_first_q = np.array([500_000, 1_000_000], dtype=np.int64)
    V_cached_q, I_cached_nA = ensure_curve_cached(
        cache_file=cache_file,
        group_path="ha_sym/curves/example",
        attrs=attrs,
        V_requested_q=V_first_q,
        evaluate_missing_q=evaluate_missing_q,
    )
    np.testing.assert_array_equal(V_cached_q, V_first_q)
    np.testing.assert_allclose(I_cached_nA, V_first_q.astype(np.float64))
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], V_first_q)

    V_second_q = np.array(
        [100_000, 200_000, 500_000, 1_000_000],
        dtype=np.int64,
    )
    V_cached_q, I_cached_nA = ensure_curve_cached(
        cache_file=cache_file,
        group_path="ha_sym/curves/example",
        attrs=attrs,
        V_requested_q=V_second_q,
        evaluate_missing_q=evaluate_missing_q,
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(
        calls[1],
        np.array([100_000, 200_000], dtype=np.int64),
    )
    np.testing.assert_array_equal(V_cached_q, V_second_q)
    np.testing.assert_allclose(I_cached_nA, V_second_q.astype(np.float64))
