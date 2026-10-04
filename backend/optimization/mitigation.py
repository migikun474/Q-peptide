"""Error mitigation.

The noise study measures how badly results degrade. This module is the other half of that
sentence: what is done about it.

Two mechanisms, chosen because they apply to this circuit family and are measurable:

1. **Readout-error mitigation** (classical, works on simulator and hardware alike).
   Measurement errors are estimated per qubit and inverted. A full 2^n x 2^n confusion
   matrix is impossible at 16-20 qubits, so the standard *tensored* assumption is used:
   readout errors are independent across qubits, so the confusion matrix factorises and
   its inverse can be applied one qubit at a time in O(n 2^n).

2. **Dynamical decoupling + twirling** (hardware only, via Qiskit Runtime options).
   DD suppresses idle-time dephasing, which matters here because the cost layer leaves
   many qubits idle while distant pairs interact. Twirling converts coherent gate error
   into stochastic error, which readout mitigation and averaging handle far better.

The tensored assumption is an approximation: it ignores correlated readout crosstalk. It
is stated rather than hidden, and the measured improvement is reported either way.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.optimization.ising import bitstring_to_x


@dataclass
class ReadoutCalibration:
    """Per-qubit readout error rates.

    ``p01[i]`` is P(measure 1 | prepared 0) and ``p10[i]`` is P(measure 0 | prepared 1)
    for qubit ``i``.
    """
    n_qubits: int
    p01: np.ndarray
    p10: np.ndarray
    shots: int
    source: str

    def as_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "shots_per_calibration_circuit": self.shots,
            "source": self.source,
            "mean_p01": float(np.mean(self.p01)),
            "mean_p10": float(np.mean(self.p10)),
            "max_p01": float(np.max(self.p01)),
            "max_p10": float(np.max(self.p10)),
            "assumption": (
                "readout errors are independent across qubits (tensored model); this "
                "ignores correlated readout crosstalk"
            ),
        }

    def is_degenerate(self, tol: float = 1e-9) -> bool:
        """True when any qubit's 2x2 confusion matrix is not invertible."""
        det = (1.0 - self.p01) * (1.0 - self.p10) - self.p01 * self.p10
        return bool(np.any(np.abs(det) < tol))


def calibrate_readout(
    n_qubits: int,
    backend=None,
    noise_model=None,
    shots: int = 4096,
    seed: int = 0,
) -> ReadoutCalibration:
    """Estimate per-qubit readout error from two calibration circuits.

    Preparing all-zeros and all-ones and measuring gives both error directions for every
    qubit at once: two circuits regardless of ``n_qubits``, rather than 2^n.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator

    if backend is None:
        backend = (
            AerSimulator(noise_model=noise_model, seed_simulator=seed)
            if noise_model is not None
            else AerSimulator(seed_simulator=seed)
        )

    def _counts(tqc) -> dict[str, int]:
        """Execute one calibration circuit.

        Aer still exposes ``backend.run``; IBM Runtime backends have removed it in favour
        of the primitives, so the Sampler path is used there. Trying ``run`` first keeps
        the fast local path for simulators without a capability probe.
        """
        try:
            return backend.run(tqc, shots=shots).result().get_counts()
        except Exception:
            from qiskit_ibm_runtime import SamplerV2

            sampler = SamplerV2(mode=backend)
            # Calibration must measure the device's RAW readout behaviour. Twirling the
            # measurement here would average away the very asymmetry being estimated.
            try:
                sampler.options.twirling.enable_measure = False
            except Exception:
                pass
            result = sampler.run([tqc], shots=shots).result()
            data = result[0].data
            field = next(iter(data.keys())) if hasattr(data, "keys") else "meas"
            return getattr(data, field).get_counts()

    def run(prepare_ones: bool) -> np.ndarray:
        qc = QuantumCircuit(n_qubits)
        if prepare_ones:
            qc.x(range(n_qubits))
        qc.measure_all()
        tqc = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=seed)
        counts = _counts(tqc)
        # fraction of shots in which each qubit came back as 1
        ones = np.zeros(n_qubits, dtype=float)
        total = sum(counts.values())
        for key, c in counts.items():
            x = bitstring_to_x(key, n_qubits)
            ones += c * x
        return ones / max(total, 1)

    frac_ones_from_zeros = run(prepare_ones=False)   # should be ~0  -> this IS p01
    frac_ones_from_ones = run(prepare_ones=True)     # should be ~1  -> p10 = 1 - this

    name = getattr(backend, "name", str(backend))
    if callable(name):
        name = name()

    return ReadoutCalibration(
        n_qubits=n_qubits,
        p01=np.clip(frac_ones_from_zeros, 0.0, 0.49),
        p10=np.clip(1.0 - frac_ones_from_ones, 0.0, 0.49),
        shots=shots,
        source=f"two calibration circuits (|0..0>, |1..1>) on {name}",
    )


def _apply_single_qubit_matrix(vec: np.ndarray, qubit: int, m: np.ndarray) -> np.ndarray:
    """Apply a 2x2 matrix along one qubit's axis of a 2^n probability vector.

    Same reshape trick the statevector simulator uses, which keeps this O(2^n) per qubit
    instead of materialising a 2^n x 2^n operator.
    """
    dim = vec.size
    stride = 1 << qubit
    arr = vec.reshape(dim // (2 * stride), 2, stride)
    a0 = arr[:, 0, :].copy()
    a1 = arr[:, 1, :].copy()
    arr[:, 0, :] = m[0, 0] * a0 + m[0, 1] * a1
    arr[:, 1, :] = m[1, 0] * a0 + m[1, 1] * a1
    return arr.reshape(dim)


def mitigate_probabilities(
    probs: np.ndarray, cal: ReadoutCalibration, clip_negatives: bool = True
) -> np.ndarray:
    """Invert the tensored readout-confusion model.

    For each qubit the 2x2 confusion matrix is

        A = [[1 - p01,     p10],
             [    p01, 1 - p10]]       (column = prepared, row = measured)

    and the correction applies ``A^-1`` along that qubit's axis. Inversion can produce
    small negative quasi-probabilities; they are clipped and the vector renormalised,
    which is the standard pragmatic choice.
    """
    out = np.array(probs, dtype=np.float64, copy=True)
    for q in range(cal.n_qubits):
        e01, e10 = cal.p01[q], cal.p10[q]
        det = (1.0 - e01) * (1.0 - e10) - e01 * e10
        if abs(det) < 1e-12:
            continue  # non-invertible for this qubit; leave it uncorrected
        inv = np.array([[1.0 - e10, -e10], [-e01, 1.0 - e01]]) / det
        out = _apply_single_qubit_matrix(out, q, inv)

    if clip_negatives:
        out = np.clip(out, 0.0, None)
    total = out.sum()
    return out / total if total > 0 else probs


def counts_to_probabilities(counts: dict[str, int], n_qubits: int) -> np.ndarray:
    """Measurement counts -> a dense 2^n probability vector in little-endian order."""
    probs = np.zeros(1 << n_qubits, dtype=np.float64)
    total = sum(counts.values())
    if total == 0:
        return probs
    for key, c in counts.items():
        x = bitstring_to_x(key, n_qubits)
        probs[int(sum(int(v) << b for b, v in enumerate(x)))] += c / total
    return probs


def apply_runtime_mitigation_options(sampler, dynamical_decoupling: bool = True,
                                     twirling: bool = True) -> dict:
    """Enable Qiskit Runtime's built-in mitigation on a SamplerV2.

    Returns a record of what was actually enabled -- option names have moved between
    Runtime releases, so each is set defensively and the outcome reported rather than
    assumed.
    """
    applied: dict = {"dynamical_decoupling": False, "twirling": False, "errors": []}
    if dynamical_decoupling:
        try:
            sampler.options.dynamical_decoupling.enable = True
            sampler.options.dynamical_decoupling.sequence_type = "XY4"
            applied["dynamical_decoupling"] = True
            applied["dd_sequence"] = "XY4"
        except Exception as exc:
            applied["errors"].append(f"dynamical_decoupling: {type(exc).__name__}: {exc}")
    if twirling:
        try:
            sampler.options.twirling.enable_gates = True
            sampler.options.twirling.enable_measure = True
            applied["twirling"] = True
        except Exception as exc:
            applied["errors"].append(f"twirling: {type(exc).__name__}: {exc}")
    return applied


def evaluate_mitigation(
    problem,
    p: int,
    params: np.ndarray,
    noise_model,
    diag: np.ndarray,
    shots: int = 8192,
    seed: int = 0,
    calibration_shots: int = 4096,
) -> dict:
    """Measure what readout mitigation actually buys, on the same noisy circuit.

    Returns the raw and mitigated statistics side by side so the improvement (or absence
    of one) is visible rather than asserted.
    """
    from backend.optimization.qaoa import _probabilities_of_interest, sample_distribution

    n = problem.n_vars

    raw = sample_distribution(problem, p, params, shots=shots,
                              noise_model=noise_model, seed=seed)
    raw_probs = counts_to_probabilities(raw["distribution_counts"], n) \
        if "distribution_counts" in raw else _probs_from_fractions(raw["distribution"], n)

    cal = calibrate_readout(n, noise_model=noise_model, shots=calibration_shots, seed=seed)
    mit_probs = mitigate_probabilities(raw_probs, cal)

    raw_succ, raw_feas = _probabilities_of_interest(problem, raw_probs, diag, None, n)
    mit_succ, mit_feas = _probabilities_of_interest(problem, mit_probs, diag, None, n)

    return {
        "status": "evaluated",
        "shots": shots,
        "calibration": cal.as_dict(),
        "raw": {
            "expectation": float(np.dot(raw_probs, diag)),
            "success_probability": raw_succ,
            "feasible_probability": raw_feas,
        },
        "mitigated": {
            "expectation": float(np.dot(mit_probs, diag)),
            "success_probability": mit_succ,
            "feasible_probability": mit_feas,
        },
        "improvement": {
            "success_probability_ratio": (mit_succ / raw_succ) if raw_succ > 0 else None,
            "feasible_probability_delta": mit_feas - raw_feas,
            "expectation_delta": float(np.dot(mit_probs, diag) - np.dot(raw_probs, diag)),
        },
        "method": "tensored readout-error inversion (per-qubit confusion matrices)",
        "caveat": (
            "readout mitigation corrects MEASUREMENT error only. It cannot undo gate "
            "error accumulated during the circuit, which dominates at this two-qubit "
            "gate count"
        ),
    }


def _probs_from_fractions(dist: dict[str, float], n_qubits: int) -> np.ndarray:
    probs = np.zeros(1 << n_qubits, dtype=np.float64)
    for key, f in dist.items():
        x = bitstring_to_x(key, n_qubits)
        probs[int(sum(int(v) << b for b, v in enumerate(x)))] += f
    total = probs.sum()
    return probs / total if total > 0 else probs
