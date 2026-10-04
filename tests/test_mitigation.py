"""Tests for error mitigation."""
from __future__ import annotations

import numpy as np
import pytest

from backend.optimization.mitigation import (
    ReadoutCalibration,
    calibrate_readout,
    counts_to_probabilities,
    mitigate_probabilities,
)


def _cal(n: int, p01: float, p10: float) -> ReadoutCalibration:
    return ReadoutCalibration(
        n_qubits=n,
        p01=np.full(n, p01),
        p10=np.full(n, p10),
        shots=1000,
        source="synthetic",
    )


# --- the inversion itself ---------------------------------------------------
def test_zero_error_calibration_is_identity():
    """With no readout error, mitigation must not alter the distribution at all."""
    rng = np.random.default_rng(0)
    probs = rng.random(16)
    probs /= probs.sum()
    out = mitigate_probabilities(probs, _cal(4, 0.0, 0.0))
    assert np.allclose(out, probs, atol=1e-12)


def test_mitigation_inverts_a_known_corruption():
    """Corrupt a known distribution with the forward model, then recover it."""
    n, p01, p10 = 3, 0.08, 0.05
    truth = np.zeros(1 << n)
    truth[5] = 0.7
    truth[2] = 0.3

    # forward model: measured = A @ truth, applied per qubit
    A = np.array([[1 - p01, p10], [p01, 1 - p10]])
    measured = truth.copy()
    for q in range(n):
        stride = 1 << q
        arr = measured.reshape(-1, 2, stride)
        a0, a1 = arr[:, 0, :].copy(), arr[:, 1, :].copy()
        arr[:, 0, :] = A[0, 0] * a0 + A[0, 1] * a1
        arr[:, 1, :] = A[1, 0] * a0 + A[1, 1] * a1
        measured = arr.reshape(1 << n)

    assert not np.allclose(measured, truth, atol=1e-6), "corruption had no effect"
    recovered = mitigate_probabilities(measured, _cal(n, p01, p10))
    assert np.allclose(recovered, truth, atol=1e-8)


def test_output_is_a_normalised_probability_distribution():
    rng = np.random.default_rng(1)
    probs = rng.random(32)
    probs /= probs.sum()
    out = mitigate_probabilities(probs, _cal(5, 0.05, 0.07))
    assert out.sum() == pytest.approx(1.0)
    assert np.all(out >= 0.0), "clipping must leave no negative quasi-probabilities"


def test_degenerate_calibration_is_detected_and_skipped():
    """A 50/50 confusion matrix carries no information and must not blow up."""
    cal = _cal(3, 0.5, 0.5)
    assert cal.is_degenerate()
    probs = np.full(8, 1 / 8)
    out = mitigate_probabilities(probs, cal)
    assert np.all(np.isfinite(out))
    assert out.sum() == pytest.approx(1.0)


def test_asymmetric_errors_are_handled_per_direction():
    """p01 and p10 are different physical processes and must not be conflated."""
    n = 2
    cal = ReadoutCalibration(n, np.array([0.10, 0.02]), np.array([0.01, 0.15]), 1000, "x")
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    out = mitigate_probabilities(probs, cal)
    assert out.sum() == pytest.approx(1.0)
    assert not np.allclose(out, probs)


# --- counts conversion ------------------------------------------------------
def test_counts_to_probabilities_little_endian():
    # "01" means qubit1=0, qubit0=1 -> index 1
    probs = counts_to_probabilities({"01": 10}, 2)
    assert probs[1] == pytest.approx(1.0)
    probs = counts_to_probabilities({"10": 10}, 2)
    assert probs[2] == pytest.approx(1.0)


def test_counts_to_probabilities_normalises():
    probs = counts_to_probabilities({"00": 30, "11": 10}, 2)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.75)
    assert probs[3] == pytest.approx(0.25)


def test_empty_counts_do_not_crash():
    assert counts_to_probabilities({}, 3).sum() == pytest.approx(0.0)


# --- calibration against a known injected error -----------------------------
def test_calibration_recovers_injected_readout_error():
    """The estimator must measure back the readout error that was injected.

    This is the check that makes the whole module trustworthy: if calibration were
    wrong, mitigation would confidently apply the wrong correction.
    """
    from backend.optimization.hardware import build_depolarizing_noise

    injected = 0.04
    noise, _ = build_depolarizing_noise(
        one_qubit_error=0.0, two_qubit_error=0.0, readout_error=injected
    )
    cal = calibrate_readout(4, noise_model=noise, shots=8192, seed=0)

    assert cal.p01.mean() == pytest.approx(injected, abs=0.012)
    assert cal.p10.mean() == pytest.approx(injected, abs=0.012)
    assert not cal.is_degenerate()


def test_noiseless_calibration_finds_no_error():
    cal = calibrate_readout(3, noise_model=None, shots=2048, seed=0)
    assert cal.p01.max() < 1e-9
    assert cal.p10.max() < 1e-9
