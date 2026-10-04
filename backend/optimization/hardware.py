"""Noise models and optional IBM Quantum hardware execution.

Noise study (spec section 33). Two documented noise models:

  ``depolarizing``  a parameterised depolarizing + readout-error model whose rates are
                    stated explicitly, so the study is reproducible without network
                    access and without pinning a particular device calibration.
  ``device``        a model built from a real backend's live calibration data via
                    ``NoiseModel.from_backend``. Requires credentials; the backend name
                    and the calibration timestamp are recorded.

Hardware (spec section 34). Credentials are read from the environment only -- never from
source, never written to a results file. If no token is present every function here
degrades to a clear "not evaluated" record and the simulator pipeline continues to work.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

# Default rates. These are representative of current superconducting hardware rather than
# a specific device; they are parameters of the study, reported with every result.
DEFAULT_ONE_QUBIT_ERROR = 3.0e-4
DEFAULT_TWO_QUBIT_ERROR = 1.0e-2
DEFAULT_READOUT_ERROR = 1.5e-2


@dataclass
class NoiseSpec:
    """A fully described noise model, so a run can be reproduced from the record."""
    name: str
    one_qubit_error: float
    two_qubit_error: float
    readout_error: float
    description: str

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "one_qubit_depolarizing_error": self.one_qubit_error,
            "two_qubit_depolarizing_error": self.two_qubit_error,
            "readout_error": self.readout_error,
            "description": self.description,
        }


def build_depolarizing_noise(
    one_qubit_error: float = DEFAULT_ONE_QUBIT_ERROR,
    two_qubit_error: float = DEFAULT_TWO_QUBIT_ERROR,
    readout_error: float = DEFAULT_READOUT_ERROR,
    name: str = "depolarizing_representative",
) -> tuple[NoiseModel, NoiseSpec]:
    """Depolarizing gate noise plus symmetric readout error.

    Two-qubit error is set roughly 30x the one-qubit rate, which is the usual ratio on
    superconducting hardware and the reason two-qubit gate counts are reported separately
    from depth.
    """
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(one_qubit_error, 1), ["rz", "sx", "x", "h", "rx", "ry"]
    )
    model.add_all_qubit_quantum_error(
        depolarizing_error(two_qubit_error, 2), ["cx", "cz", "ecr", "rzz"]
    )
    if readout_error > 0:
        model.add_all_qubit_readout_error(
            ReadoutError(
                [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
            )
        )
    spec = NoiseSpec(
        name=name,
        one_qubit_error=one_qubit_error,
        two_qubit_error=two_qubit_error,
        readout_error=readout_error,
        description=(
            "all-qubit depolarizing error on one- and two-qubit gates plus symmetric "
            "readout error; rates are representative of current superconducting "
            "hardware, not a specific device calibration"
        ),
    )
    return model, spec


def build_thermal_noise(
    t1_us: float = 150.0,
    t2_us: float = 100.0,
    gate_time_1q_ns: float = 35.0,
    gate_time_2q_ns: float = 450.0,
    readout_error: float = DEFAULT_READOUT_ERROR,
) -> tuple[NoiseModel, NoiseSpec]:
    """Thermal-relaxation (T1/T2) noise, an alternative mechanism to depolarizing.

    Included so the noise study does not rest on a single error channel: relaxation
    penalises long circuits specifically, which is the behaviour that matters for the
    depth study.
    """
    model = NoiseModel()
    t1 = t1_us * 1e3  # ns
    t2 = min(t2_us * 1e3, 2 * t1)
    err_1q = thermal_relaxation_error(t1, t2, gate_time_1q_ns)
    err_2q = thermal_relaxation_error(t1, t2, gate_time_2q_ns).expand(
        thermal_relaxation_error(t1, t2, gate_time_2q_ns)
    )
    model.add_all_qubit_quantum_error(err_1q, ["rz", "sx", "x", "h", "rx", "ry"])
    model.add_all_qubit_quantum_error(err_2q, ["cx", "cz", "ecr", "rzz"])
    if readout_error > 0:
        model.add_all_qubit_readout_error(
            ReadoutError(
                [[1 - readout_error, readout_error], [readout_error, 1 - readout_error]]
            )
        )
    spec = NoiseSpec(
        name="thermal_relaxation",
        one_qubit_error=gate_time_1q_ns / t1,
        two_qubit_error=gate_time_2q_ns / t1,
        readout_error=readout_error,
        description=(
            f"T1={t1_us}us, T2={t2_us}us thermal relaxation with "
            f"{gate_time_1q_ns}ns/{gate_time_2q_ns}ns gate times plus readout error"
        ),
    )
    return model, spec


def noise_sweep_specs() -> list[tuple[str, float]]:
    """Two-qubit error rates for the noise-strength sweep."""
    return [
        ("noiseless", 0.0),
        ("low_noise", 1.0e-3),
        ("representative", 1.0e-2),
        ("high_noise", 3.0e-2),
    ]


# ---------------------------------------------------------------------------
# IBM Quantum
# ---------------------------------------------------------------------------
IBM_TOKEN_VARS = ("QISKIT_IBM_TOKEN", "IBM_QUANTUM_TOKEN")
IBM_INSTANCE_VARS = ("QISKIT_IBM_INSTANCE", "IBM_QUANTUM_INSTANCE")


def ibm_credentials_available() -> bool:
    """True when an IBM Quantum token is present in the environment."""
    return any(os.environ.get(v) for v in IBM_TOKEN_VARS)


def _token() -> str | None:
    for v in IBM_TOKEN_VARS:
        val = os.environ.get(v)
        if val:
            return val
    return None


def _instance() -> str | None:
    for v in IBM_INSTANCE_VARS:
        val = os.environ.get(v)
        if val:
            return val
    return None


def get_ibm_service(explain: bool = False):
    """A ``QiskitRuntimeService`` from environment credentials, or None.

    Never persists or echoes the token. With ``explain=True`` returns
    ``(service, diagnostic)`` so a failed connection can say *why* -- a wrong key and a
    missing instance CRN fail very differently, and silently returning None makes them
    indistinguishable.
    """
    diag: dict = {"token_present": ibm_credentials_available(),
                  "instance_present": _instance() is not None,
                  "attempts": []}

    if not diag["token_present"]:
        diag["error"] = (
            f"no API key in the environment (looked for {', '.join(IBM_TOKEN_VARS)})"
        )
        return (None, diag) if explain else None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as exc:
        diag["error"] = f"qiskit-ibm-runtime not installed: {exc}"
        return (None, diag) if explain else None

    kwargs = {"token": _token()}
    inst = _instance()
    if inst:
        kwargs["instance"] = inst

    # The current IBM Quantum Platform uses IBM Cloud IAM keys and an instance CRN.
    # The legacy channel is tried second so older accounts still work.
    for channel in ("ibm_quantum_platform", None):
        label = channel or "default"
        try:
            svc = (QiskitRuntimeService(channel=channel, **kwargs) if channel
                   else QiskitRuntimeService(**kwargs))
            diag["attempts"].append({"channel": label, "ok": True})
            diag["channel_used"] = label
            return (svc, diag) if explain else svc
        except Exception as exc:
            diag["attempts"].append({"channel": label, "ok": False,
                                     "error": f"{type(exc).__name__}: {exc}"})

    diag["error"] = "all channels failed; see attempts"
    if not diag["instance_present"]:
        diag["hint"] = (
            "no instance CRN set. The current IBM Quantum Platform requires one: set "
            "QISKIT_IBM_INSTANCE to the CRN of your Qiskit Runtime service instance "
            "(IBM Cloud -> Resource list -> your Quantum service -> Details -> CRN)."
        )
    return (None, diag) if explain else None


def select_backend(service, min_qubits: int, backend_name: str | None = None):
    """Least-busy operational backend with enough qubits, or a named one."""
    if service is None:
        return None
    try:
        if backend_name:
            return service.backend(backend_name)
        return service.least_busy(operational=True, simulator=False,
                                 min_num_qubits=min_qubits)
    except Exception:
        return None


def device_noise_model(backend) -> tuple[NoiseModel | None, dict]:
    """Noise model from a real backend's calibration, with provenance."""
    if backend is None:
        return None, {"status": "not evaluated", "reason": "no backend available"}
    try:
        model = NoiseModel.from_backend(backend)
        name = getattr(backend, "name", str(backend))
        if callable(name):
            name = name()
        return model, {
            "status": "built",
            "backend": name,
            "n_qubits": getattr(backend, "num_qubits", None),
            "source": "qiskit_aer.noise.NoiseModel.from_backend (live calibration)",
        }
    except Exception as exc:
        return None, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def run_on_hardware(
    problem,
    p: int,
    params,
    shots: int = 4096,
    backend_name: str | None = None,
    optimization_level: int = 3,
) -> dict:
    """Execute the final QAOA circuit on real IBM hardware, if available.

    Records backend name, job id, shots, transpiled depth, two-qubit gate count and the
    measured distribution. Returns a ``not evaluated`` record when no QPU is reachable --
    never a fabricated result.
    """
    import numpy as np

    from backend.optimization.ising import bitstring_to_x, diagonal_energies, qubo_to_ising
    from backend.optimization.qaoa import (
        _probabilities_of_interest,
        count_two_qubit_gates,
        final_circuit,
    )

    if not ibm_credentials_available():
        return {
            "status": "not evaluated",
            "reason": (
                "no IBM Quantum token in the environment "
                f"(looked for {', '.join(IBM_TOKEN_VARS)})"
            ),
        }
    service = get_ibm_service()
    if service is None:
        return {"status": "not evaluated",
                "reason": "could not initialise QiskitRuntimeService"}

    backend = select_backend(service, problem.n_vars, backend_name)
    if backend is None:
        return {"status": "not evaluated",
                "reason": f"no operational backend with >= {problem.n_vars} qubits"}

    name = getattr(backend, "name", str(backend))
    if callable(name):
        name = name()

    try:
        from qiskit import transpile
        from qiskit_ibm_runtime import SamplerV2

        circuit = final_circuit(problem, p, params, measure=True)
        tqc = transpile(circuit, backend=backend,
                        optimization_level=optimization_level, seed_transpiler=0)

        sampler = SamplerV2(mode=backend)
        # Hardware-side mitigation. DD suppresses idle dephasing, which matters here
        # because the cost layer leaves many qubits idle while distant pairs interact;
        # twirling turns coherent gate error into stochastic error, which averaging and
        # readout correction handle far better.
        from backend.optimization.mitigation import apply_runtime_mitigation_options
        mitigation_applied = apply_runtime_mitigation_options(sampler)

        job = sampler.run([tqc], shots=shots)
        job_id = job.job_id()
        result = job.result()
        counts = result[0].data.meas.get_counts()

        n = problem.n_vars
        hamiltonian, _ = qubo_to_ising(problem)
        diag = diagonal_energies(hamiltonian, n)
        total = sum(counts.values())
        probs = np.zeros(1 << n, dtype=float)
        for key, c in counts.items():
            x = bitstring_to_x(key, n)
            probs[int(sum(int(v) << b for b, v in enumerate(x)))] += c / total

        success, feasible = _probabilities_of_interest(problem, probs, diag, None, n)

        # Classical readout mitigation, calibrated on the SAME backend.
        readout_record: dict = {"status": "not evaluated"}
        try:
            from backend.optimization.mitigation import (
                calibrate_readout, mitigate_probabilities,
            )
            cal = calibrate_readout(n, backend=backend, shots=min(shots, 2048))
            mit_probs = mitigate_probabilities(probs, cal)
            mit_success, mit_feasible = _probabilities_of_interest(
                problem, mit_probs, diag, None, n
            )
            readout_record = {
                "status": "evaluated",
                "calibration": cal.as_dict(),
                "expectation": float(np.dot(mit_probs, diag)),
                "success_probability": mit_success,
                "feasible_probability": mit_feasible,
            }
        except Exception as exc:
            readout_record = {"status": "failed",
                              "error": f"{type(exc).__name__}: {exc}"}

        sampled = np.flatnonzero(probs > 0)
        best_code = int(sampled[np.argmin(diag[sampled])])
        best_x = np.array([(best_code >> b) & 1 for b in range(n)], dtype=int)
        t_ops = dict(tqc.count_ops())

        return {
            "status": "executed",
            "backend": name,
            "job_id": job_id,
            "shots": shots,
            "p": p,
            "logical_depth": circuit.depth(),
            "transpiled_depth": tqc.depth(),
            "transpiled_size": tqc.size(),
            "transpiled_ops": t_ops,
            "transpiled_two_qubit_gates": count_two_qubit_gates(tqc),
            "two_qubit_gates_counted_by": "gate arity (basis-independent)",
            "optimization_level": optimization_level,
            "expectation": float(np.dot(probs, diag)),
            "best_sampled_energy": float(diag[best_code]),
            "best_sampled_x": best_x.tolist(),
            "best_sampled_feasible": bool(problem.is_feasible(best_x)),
            "success_probability": success,
            "feasible_probability": feasible,
            "n_distinct_outcomes": len(counts),
            "distribution": {k: v / total for k, v in counts.items()},
            "mitigation": {
                "runtime_options": mitigation_applied,
                "readout_mitigation": readout_record,
                "note": (
                    "runtime options are applied at execution; readout mitigation is "
                    "applied classically afterwards using a calibration measured on the "
                    "same backend. Neither can undo gate error accumulated during the "
                    "circuit."
                ),
            },
        }
    except Exception as exc:
        return {
            "status": "failed",
            "backend": name,
            "error": f"{type(exc).__name__}: {exc}",
            "note": "hardware execution failed; no result is reported",
        }
