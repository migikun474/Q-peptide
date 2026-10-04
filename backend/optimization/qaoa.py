"""QAOA for the mutation-selection QUBO, on ideal and noisy simulators and optional QPU.

Ansatz (spec section 27):

    |psi(gamma,beta)> = prod_{l=1..p} exp(-i beta_l sum_i X_i) exp(-i gamma_l H_C) |+>^N

The standard transverse-field X mixer is used with the penalty-based QUBO, as permitted
for the MVP; a constraint-preserving mixer is the documented alternative (see
research/literature_review.md section 7.2).

TWO EQUIVALENT COST-LAYER IMPLEMENTATIONS
Because ``H_C`` is diagonal (only I, Z and ZZ terms), ``exp(-i gamma H_C)`` factorises
exactly into commuting one- and two-qubit rotations:

    exp(-i gamma h_i Z_i)       = RZ(2 gamma h_i)  on qubit i
    exp(-i gamma J_ij Z_i Z_j)  = RZZ(2 gamma J_ij) on (i, j)

so the circuit is built explicitly from RZ/RZZ/RX rather than from a generic Pauli
evolution gate. This is both exact and far cheaper: generic synthesis exponentiates a
sparse matrix on every objective evaluation, which dominates the runtime.

For the ideal-simulator optimisation loop the same state is produced by a small
specialised NumPy simulator (:func:`exact_statevector`), which applies the diagonal phase
in one vectorised multiply and each RX by a reshape. ``verify_simulator_agreement``
asserts it matches Qiskit's own ``Statevector`` on the explicit circuit, so the fast path
is checked rather than trusted.

Variational objective (spec section 28): ``F(gamma,beta) = <psi|H_C|psi>``. On the ideal
simulator F is computed EXACTLY, so the depth study measures the ansatz rather than a
shot budget. The noisy and hardware paths necessarily sample, and say so.

QUBO energy and biological score are different quantities related by the documented sign
and penalty transformation. Nothing here interprets a bitstring biologically -- decoding
and independent re-scoring happen in the experiment layer.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from backend.optimization.ising import (
    bitstring_to_x,
    diagonal_energies,
    qubo_to_ising,
    x_to_bitstring,
)
from backend.optimization.qubo import QuboProblem

SUPPORTED_OPTIMIZERS = ("COBYLA", "Nelder-Mead", "Powell")

TWO_QUBIT_GATE_NAMES = {"cx", "cz", "ecr", "cp", "rzz", "swap", "iswap"}
HARDWARE_TWO_QUBIT_NAMES = {"cx", "cz", "ecr", "swap", "iswap"}


def count_two_qubit_gates(circuit: QuantumCircuit) -> int:
    """Number of instructions acting on two or more qubits.

    Counted by gate ARITY rather than by name. A name-based count is basis-dependent and
    silently wrong: Aer supports ``rzz`` natively, so a circuit transpiled for it has
    zero ``cx`` gates while still carrying every entangling operation. Arity is the
    quantity that actually drives hardware error, so it is what gets reported.

    Barriers and measurements are excluded -- they are not entangling operations.
    """
    excluded = {"barrier", "measure", "delay", "reset", "snapshot"}
    total = 0
    for instruction in circuit.data:
        op = instruction.operation
        if op.name in excluded:
            continue
        if len(instruction.qubits) >= 2:
            total += 1
    return total


# ---------------------------------------------------------------------------
# Ising coefficient extraction
# ---------------------------------------------------------------------------
@dataclass
class IsingCoefficients:
    """Decomposition of a diagonal Hamiltonian into offset, fields and couplings."""
    n_qubits: int
    offset: float
    h: np.ndarray                       # shape (n,), coefficient of Z_i
    J: dict[tuple[int, int], float]     # coefficient of Z_i Z_j, i < j

    def max_abs(self) -> float:
        vals = [abs(v) for v in self.J.values()] + [float(np.max(np.abs(self.h)))
                                                    if self.n_qubits else 0.0]
        return max(vals) if vals else 0.0


def extract_ising_coefficients(hamiltonian: SparsePauliOp) -> IsingCoefficients:
    """Pull offset / h / J out of a diagonal SparsePauliOp, honouring bit order."""
    n = hamiltonian.num_qubits
    offset = 0.0
    h = np.zeros(n, dtype=float)
    J: dict[tuple[int, int], float] = {}

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = str(pauli)
        c = float(np.real(coeff))
        qubits = [n - 1 - pos for pos, ch in enumerate(label) if ch == "Z"]
        if any(ch not in "IZ" for ch in label):
            raise ValueError("cost Hamiltonian must be diagonal (I/Z only)")
        if not qubits:
            offset += c
        elif len(qubits) == 1:
            h[qubits[0]] += c
        elif len(qubits) == 2:
            a, b = sorted(qubits)
            J[(a, b)] = J.get((a, b), 0.0) + c
        else:
            raise ValueError(
                f"cost Hamiltonian has a {len(qubits)}-body term; QUBO Hamiltonians are "
                "at most 2-body"
            )
    return IsingCoefficients(n, offset, h, J)


# ---------------------------------------------------------------------------
# Explicit circuit
# ---------------------------------------------------------------------------
def build_qaoa_circuit(
    coeffs: IsingCoefficients, p: int, scale: float = 1.0
) -> tuple[QuantumCircuit, list[Parameter], list[Parameter]]:
    """Explicit QAOA circuit from RZ / RZZ / RX gates.

    ``scale`` divides the cost coefficients used in the circuit. Penalised QUBOs have
    very large coefficients, so ``exp(-i gamma H_C)`` wraps through many full rotations
    for any reasonable ``gamma``, making the parameter landscape violently oscillatory.
    Dividing by the largest coefficient is a pure reparameterisation of ``gamma``: it
    changes neither the spectrum ordering nor any reported energy, since energies are
    always computed from the unscaled diagonal.

    Parameter order is ``[gamma_0, beta_0, gamma_1, beta_1, ...]``.
    """
    n = coeffs.n_qubits
    qc = QuantumCircuit(n, name=f"QAOA_p{p}")
    qc.h(range(n))

    gammas = [Parameter(f"gamma_{l}") for l in range(p)]
    betas = [Parameter(f"beta_{l}") for l in range(p)]

    s = scale if scale not in (0.0,) else 1.0
    for l in range(p):
        g = gammas[l]
        # single-qubit phases: exp(-i gamma h_i Z_i) = RZ(2 gamma h_i)
        for i in range(n):
            hi = coeffs.h[i] / s
            if abs(hi) > 1e-15:
                qc.rz(2.0 * g * hi, i)
        # couplings: exp(-i gamma J_ij Z_i Z_j) = RZZ(2 gamma J_ij)
        for (i, j), Jij in sorted(coeffs.J.items()):
            Js = Jij / s
            if abs(Js) > 1e-15:
                qc.rzz(2.0 * g * Js, i, j)
        qc.barrier()
        # mixer: exp(-i beta X_i) = RX(2 beta)
        for i in range(n):
            qc.rx(2.0 * betas[l], i)
        if l < p - 1:
            qc.barrier()

    ordered: list[Parameter] = []
    for l in range(p):
        ordered += [gammas[l], betas[l]]
    return qc, gammas, betas


def bind(qc: QuantumCircuit, p: int, theta: np.ndarray) -> QuantumCircuit:
    """Bind ``[gamma_0, beta_0, ...]`` by parameter name, independent of internal order."""
    theta = np.asarray(theta, dtype=float)
    mapping = {}
    for param in qc.parameters:
        name = param.name
        kind, idx = name.rsplit("_", 1)
        l = int(idx)
        mapping[param] = float(theta[2 * l] if kind == "gamma" else theta[2 * l + 1])
    return qc.assign_parameters(mapping)


# ---------------------------------------------------------------------------
# Fast exact simulator
# ---------------------------------------------------------------------------
def exact_statevector(
    coeffs: IsingCoefficients,
    p: int,
    theta: np.ndarray,
    scale: float = 1.0,
    diag: np.ndarray | None = None,
) -> np.ndarray:
    """Statevector of the QAOA circuit, computed directly.

    Index convention matches Qiskit: basis index ``code`` has qubit ``i`` equal to
    ``(code >> i) & 1``.
    """
    n = coeffs.n_qubits
    dim = 1 << n
    theta = np.asarray(theta, dtype=float)

    # diagonal of the scaled, offset-free cost operator. Passing it in avoids
    # rebuilding a (2^n, n) bit matrix on every call, which otherwise dominates the
    # runtime of the variational loop.
    if diag is None:
        diag = scaled_diagonal(coeffs, scale)

    psi = np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)
    for l in range(p):
        gamma = theta[2 * l]
        beta = theta[2 * l + 1]
        psi *= np.exp(-1j * gamma * diag)
        # RX(2 beta) on every qubit: [[cos b, -i sin b], [-i sin b, cos b]]
        c, s = np.cos(beta), np.sin(beta)
        view = psi.reshape(-1)
        for i in range(n):
            stride = 1 << i
            arr = view.reshape(dim // (2 * stride), 2, stride)
            a0 = arr[:, 0, :].copy()
            a1 = arr[:, 1, :].copy()
            arr[:, 0, :] = c * a0 - 1j * s * a1
            arr[:, 1, :] = -1j * s * a0 + c * a1
        psi = view
    return psi


def scaled_diagonal(coeffs: IsingCoefficients, scale: float = 1.0) -> np.ndarray:
    """Diagonal of ``(sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j) / scale`` (no offset)."""
    n = coeffs.n_qubits
    dim = 1 << n
    codes = np.arange(dim, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(np.int8)
    z = (1 - 2 * bits.astype(np.int64)).astype(np.float64)

    s = scale if scale not in (0.0,) else 1.0
    diag = z @ (coeffs.h / s)
    for (i, j), Jij in coeffs.J.items():
        diag += (Jij / s) * z[:, i] * z[:, j]
    return diag


def verify_simulator_agreement(
    problem: QuboProblem, p: int = 2, seed: int = 0, tol: float = 1e-9
) -> dict:
    """Assert the fast NumPy simulator matches Qiskit's Statevector on the real circuit.

    This is what licenses using the fast path in the optimisation loop.
    """
    from qiskit.quantum_info import Statevector

    hamiltonian, _ = qubo_to_ising(problem)
    coeffs = extract_ising_coefficients(hamiltonian)
    scale = max(coeffs.max_abs(), 1e-12)
    qc, _, _ = build_qaoa_circuit(coeffs, p, scale)

    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=2 * p)

    fast = exact_statevector(coeffs, p, theta, scale)
    ref = np.asarray(Statevector.from_instruction(bind(qc, p, theta)).data)

    # compare up to a global phase, which no observable depends on
    overlap = np.vdot(ref, fast)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-15 else 1.0
    dev = float(np.max(np.abs(fast - phase * ref)))
    prob_dev = float(np.max(np.abs(np.abs(fast) ** 2 - np.abs(ref) ** 2)))

    return {
        "p": p,
        "n_qubits": coeffs.n_qubits,
        "max_amplitude_deviation_up_to_global_phase": dev,
        "max_probability_deviation": prob_dev,
        "tolerance": tol,
        "passed": bool(prob_dev <= tol and dev <= tol),
        "note": (
            "compared against qiskit.quantum_info.Statevector on the explicit RZ/RZZ/RX "
            "circuit; amplitudes are compared modulo a global phase"
        ),
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class QaoaResult:
    """Everything recorded for one QAOA run."""
    p: int
    optimizer: str
    initial_params: list[float]
    final_params: list[float]
    n_iterations: int
    n_function_evaluations: int
    final_expectation: float
    best_sampled_energy: float
    best_sampled_x: list[int]
    best_sampled_feasible: bool
    success_probability: float
    feasible_probability: float
    distribution: dict[str, float]
    convergence: list[float]
    circuit_metrics: dict
    backend_label: str
    shots: int | None
    runtime_seconds: float
    extra: dict = field(default_factory=dict)

    def as_dict(self, max_distribution: int = 64) -> dict:
        top = dict(
            sorted(self.distribution.items(), key=lambda kv: -kv[1])[:max_distribution]
        )
        return {
            "p": self.p,
            "optimizer": self.optimizer,
            "initial_params": self.initial_params,
            "final_params": self.final_params,
            "n_iterations": self.n_iterations,
            "n_function_evaluations": self.n_function_evaluations,
            "final_expectation": self.final_expectation,
            "best_sampled_energy": self.best_sampled_energy,
            "best_sampled_x": self.best_sampled_x,
            "best_sampled_feasible": self.best_sampled_feasible,
            "success_probability": self.success_probability,
            "feasible_probability": self.feasible_probability,
            "distribution_top": top,
            "distribution_support": len(self.distribution),
            "convergence": self.convergence,
            "circuit_metrics": self.circuit_metrics,
            "backend": self.backend_label,
            "shots": self.shots,
            "runtime_seconds": self.runtime_seconds,
            **({"extra": self.extra} if self.extra else {}),
        }


def circuit_metrics(
    qc: QuantumCircuit,
    p: int,
    params: np.ndarray | None = None,
    backend=None,
    optimization_level: int = 3,
    seed_transpiler: int = 0,
) -> dict:
    """Logical and transpiled circuit metrics.

    Both are reported because hardware cost is driven by the transpiled two-qubit gate
    count, not the logical depth (spec section 35).
    """
    bound = qc if params is None else bind(qc, p, params)
    logical_ops = dict(bound.count_ops())
    metrics = {
        "n_qubits": bound.num_qubits,
        "logical_depth": bound.depth(),
        "logical_size": bound.size(),
        "logical_ops": logical_ops,
        "logical_two_qubit_gates": count_two_qubit_gates(bound),
        "n_parameters": qc.num_parameters,
    }
    basis = ["rz", "sx", "x", "cx"]
    try:
        if backend is not None:
            t = transpile(bound, backend=backend, optimization_level=optimization_level,
                          seed_transpiler=seed_transpiler)
            label = getattr(backend, "name", str(backend))
            if callable(label):
                label = label()
        else:
            t = transpile(bound, basis_gates=basis,
                          optimization_level=optimization_level,
                          seed_transpiler=seed_transpiler)
            label = f"abstract basis {basis} (all-to-all; no routing overhead)"
        t_ops = dict(t.count_ops())
        metrics["transpiled"] = {
            "target": label,
            "optimization_level": optimization_level,
            "depth": t.depth(),
            "size": t.size(),
            "ops": t_ops,
            "two_qubit_gates": count_two_qubit_gates(t),
            "two_qubit_gates_counted_by": "gate arity (basis-independent)",
        }
    except Exception as exc:
        metrics["transpiled"] = {"error": f"{type(exc).__name__}: {exc}"}
    return metrics


def _initial_params(p: int, seed: int) -> np.ndarray:
    """Initial ``[gamma_0, beta_0, ...]`` following a linear adiabatic-like ramp.

    gamma increases and beta decreases across layers, mimicking an annealing schedule.
    This is a much better start than uniform random, which often lands on a plateau.
    """
    rng = np.random.default_rng(seed)
    params = np.empty(2 * p, dtype=float)
    for l in range(p):
        frac = (l + 1) / (p + 1)
        params[2 * l] = frac * 0.9 + rng.normal(0, 0.03)              # gamma
        params[2 * l + 1] = (1.0 - frac) * (np.pi / 3) + rng.normal(0, 0.03)  # beta
    return params


def _probabilities_of_interest(
    problem: QuboProblem,
    probs: np.ndarray,
    diag: np.ndarray,
    optimal_solutions: list[np.ndarray] | None,
    n: int,
    feasible_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """(success probability, feasible probability).

    Success probability sums over ALL degenerate optima (spec section 31), never the
    frequency of one arbitrary optimal bitstring.
    """
    if optimal_solutions is None:
        e_min = float(diag.min())
        opt_codes = np.flatnonzero(np.isclose(diag, e_min, rtol=0.0, atol=1e-9))
    else:
        opt_codes = np.array(
            [
                int(sum(int(v) << b for b, v in enumerate(np.asarray(s).astype(int))))
                for s in optimal_solutions
            ],
            dtype=int,
        )
    success = float(np.sum(probs[opt_codes])) if len(opt_codes) else 0.0
    if feasible_mask is None:
        feasible_mask = feasibility_mask(problem, n)
    feasible = float(np.sum(probs[feasible_mask]))
    return success, feasible


def feasibility_mask(problem: QuboProblem, n: int) -> np.ndarray:
    """Boolean mask over all ``2^n`` basis states marking feasible assignments."""
    dim = 1 << n
    codes = np.arange(dim, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(np.int8)
    n_mut = problem.n_mutation_vars
    counts = bits[:, :n_mut].sum(axis=1)

    ok = np.ones(dim, dtype=bool)
    for i, j in problem.landscape.mutation_set.conflict_pairs:
        ok &= ~((bits[:, i] == 1) & (bits[:, j] == 1))
    from backend.optimization.qubo import BudgetMode
    if problem.budget_mode is BudgetMode.EXACTLY_K:
        ok &= counts == problem.budget_k
    else:
        ok &= counts <= problem.budget_k
        # the slack bits must balance the budget equation for the penalty to vanish
        if problem.n_slack_vars > 0:
            weights = np.array(problem.build_report["slack_weights"], dtype=np.int64)
            slack = bits[:, n_mut: n_mut + problem.n_slack_vars].astype(np.int64) @ weights
            ok &= (counts + slack) == problem.budget_k
    return ok


def mutation_feasibility_mask(problem: QuboProblem, n: int) -> np.ndarray:
    """Mask of states whose MUTATION part is feasible, ignoring slack bookkeeping."""
    dim = 1 << n
    codes = np.arange(dim, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(np.int8)
    n_mut = problem.n_mutation_vars
    counts = bits[:, :n_mut].sum(axis=1)
    ok = np.ones(dim, dtype=bool)
    for i, j in problem.landscape.mutation_set.conflict_pairs:
        ok &= ~((bits[:, i] == 1) & (bits[:, j] == 1))
    from backend.optimization.qubo import BudgetMode
    if problem.budget_mode is BudgetMode.EXACTLY_K:
        ok &= counts == problem.budget_k
    else:
        ok &= counts <= problem.budget_k
    return ok


def run_qaoa_ideal(
    problem: QuboProblem,
    p: int = 1,
    optimizer: str = "COBYLA",
    maxiter: int = 400,
    seed: int = 0,
    optimal_solutions: list[np.ndarray] | None = None,
    scale_cost: bool = True,
    n_restarts: int = 1,
    shots: int = 8192,
) -> QaoaResult:
    """QAOA on the ideal simulator with an exact (shot-noise-free) objective.

    ``n_restarts`` runs the classical optimiser from several starting points and keeps the
    best, which matters because the QAOA landscape is non-convex and a single COBYLA run
    can stall in a local minimum.
    """
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {SUPPORTED_OPTIMIZERS}")

    n = problem.n_vars
    hamiltonian, _ = qubo_to_ising(problem)
    coeffs = extract_ising_coefficients(hamiltonian)
    scale = max(coeffs.max_abs(), 1e-12) if scale_cost else 1.0

    qc, _, _ = build_qaoa_circuit(coeffs, p, scale)
    diag_true = diagonal_energies(hamiltonian, n)   # unscaled, includes offset
    diag_scaled = scaled_diagonal(coeffs, scale)    # built once, reused every evaluation
    feas_mask = feasibility_mask(problem, n)

    convergence: list[float] = []
    n_evals = 0

    def objective(theta: np.ndarray) -> float:
        nonlocal n_evals
        n_evals += 1
        psi = exact_statevector(coeffs, p, theta, scale, diag_scaled)
        value = float(np.dot(np.abs(psi) ** 2, diag_true))
        convergence.append(value)
        return value

    t0 = time.perf_counter()
    best = None
    first_x0: np.ndarray | None = None
    for r in range(max(1, n_restarts)):
        x0 = _initial_params(p, seed + 1000 * r)
        if first_x0 is None:
            first_x0 = x0
        opts: dict = {"maxiter": maxiter}
        if optimizer == "COBYLA":
            opts["rhobeg"] = 0.4
        res = minimize(objective, x0, method=optimizer, options=opts)
        if best is None or res.fun < best.fun:
            best = res
    runtime = time.perf_counter() - t0
    assert best is not None and first_x0 is not None

    final_params = np.asarray(best.x, dtype=float)
    psi = exact_statevector(coeffs, p, final_params, scale, diag_scaled)
    probs = np.abs(psi) ** 2
    final_expectation = float(np.dot(probs, diag_true))

    threshold = 1e-12
    support = np.flatnonzero(probs > threshold)
    distribution = {
        x_to_bitstring(np.array([(int(code) >> b) & 1 for b in range(n)])): float(probs[code])
        for code in support
    }

    # "Best sampled" must come from a FINITE sample. Reading the minimum over the exact
    # statevector's support would be meaningless: an ideal state has nonzero amplitude on
    # essentially every basis state, so that minimum is always the global optimum
    # regardless of how good the parameters are. Drawing `shots` samples from the exact
    # distribution reproduces what a device with the same state would actually return.
    rng_shots = np.random.default_rng(seed + 99991)
    counts = rng_shots.multinomial(shots, probs / probs.sum())
    sampled_codes = np.flatnonzero(counts > 0)
    best_code = int(sampled_codes[np.argmin(diag_true[sampled_codes])])
    best_x = np.array([(best_code >> b) & 1 for b in range(n)], dtype=int)
    empirical = counts / float(shots)
    empirical_success, empirical_feasible = _probabilities_of_interest(
        problem, empirical, diag_true, optimal_solutions, n, feas_mask
    )

    success, feasible = _probabilities_of_interest(
        problem, probs, diag_true, optimal_solutions, n, feas_mask
    )

    metrics = circuit_metrics(qc, p, final_params)
    metrics["cost_operator_scaling"] = {
        "applied": scale != 1.0,
        "scale": scale,
        "note": (
            "gamma reparameterisation only; all reported energies use the unscaled "
            "Hamiltonian diagonal"
        ),
    }

    return QaoaResult(
        p=p,
        optimizer=optimizer,
        initial_params=first_x0.tolist(),
        final_params=final_params.tolist(),
        n_iterations=int(getattr(best, "nit", len(convergence))),
        n_function_evaluations=n_evals,
        final_expectation=final_expectation,
        best_sampled_energy=float(diag_true[best_code]),
        best_sampled_x=best_x.tolist(),
        best_sampled_feasible=bool(problem.is_feasible(best_x)),
        success_probability=success,
        feasible_probability=feasible,
        distribution=distribution,
        convergence=convergence,
        circuit_metrics=metrics,
        backend_label=(
            "ideal_statevector (exact expectation; best-energy and empirical "
            "probabilities from finite shots)"
        ),
        shots=shots,
        runtime_seconds=runtime,
        extra={
            "scipy_message": str(getattr(best, "message", "")),
            "scipy_success": bool(getattr(best, "success", False)),
            "objective": "exact <psi|H_C|psi>",
            "best_sampled_from": (
                f"{shots} shots drawn from the exact distribution; the exact "
                "statevector's full support makes an unsampled minimum meaningless"
            ),
            "empirical_success_probability": empirical_success,
            "empirical_feasible_probability": empirical_feasible,
            "n_distinct_sampled_outcomes": int(len(sampled_codes)),
            "mixer": "standard transverse-field X mixer",
            "identity_offset": coeffs.offset,
            "n_restarts": max(1, n_restarts),
            "simulator": "specialised NumPy QAOA simulator (verified against Qiskit)",
        },
    )


def final_circuit(
    problem: QuboProblem, p: int, params: np.ndarray, scale_cost: bool = True,
    measure: bool = True,
) -> QuantumCircuit:
    """The concrete, parameter-bound circuit actually executed for sampling."""
    hamiltonian, _ = qubo_to_ising(problem)
    coeffs = extract_ising_coefficients(hamiltonian)
    scale = max(coeffs.max_abs(), 1e-12) if scale_cost else 1.0
    qc, _, _ = build_qaoa_circuit(coeffs, p, scale)
    bound = bind(qc, p, np.asarray(params, dtype=float))
    if measure:
        bound.measure_all()
    return bound


def sample_distribution(
    problem: QuboProblem,
    p: int,
    params: np.ndarray,
    shots: int = 8192,
    noise_model=None,
    seed: int = 0,
    scale_cost: bool = True,
    backend=None,
    optimization_level: int = 3,
) -> dict:
    """Sample the final circuit; returns the measured distribution and statistics.

    Used for the noisy and hardware paths, and to confirm that finite sampling
    reproduces the ideal exact distribution.
    """
    from qiskit_aer import AerSimulator

    n = problem.n_vars
    hamiltonian, _ = qubo_to_ising(problem)
    bound = final_circuit(problem, p, params, scale_cost, measure=True)

    if backend is None:
        backend = (
            AerSimulator(noise_model=noise_model, seed_simulator=seed)
            if noise_model is not None
            else AerSimulator(seed_simulator=seed)
        )
    tqc = transpile(bound, backend=backend, optimization_level=optimization_level,
                    seed_transpiler=seed)
    t0 = time.perf_counter()
    counts = backend.run(tqc, shots=shots).result().get_counts()
    runtime = time.perf_counter() - t0

    diag = diagonal_energies(hamiltonian, n)
    total = sum(counts.values())
    probs = np.zeros(1 << n, dtype=float)
    for key, c in counts.items():
        x = bitstring_to_x(key, n)
        code = int(sum(int(v) << b for b, v in enumerate(x)))
        probs[code] += c / total

    success, feasible = _probabilities_of_interest(problem, probs, diag, None, n)
    sampled = np.flatnonzero(probs > 0)
    best_code = int(sampled[np.argmin(diag[sampled])])
    best_x = np.array([(best_code >> b) & 1 for b in range(n)], dtype=int)

    t_ops = dict(tqc.count_ops())
    label = getattr(backend, "name", str(backend))
    if callable(label):
        label = label()
    return {
        "distribution": {k: v / total for k, v in counts.items()},
        "shots": shots,
        "expectation": float(np.dot(probs, diag)),
        "best_sampled_energy": float(diag[best_code]),
        "best_sampled_x": best_x.tolist(),
        "best_sampled_feasible": bool(problem.is_feasible(best_x)),
        "success_probability": success,
        "feasible_probability": feasible,
        "n_distinct_outcomes": int(len(counts)),
        "runtime_seconds": runtime,
        "logical_depth": bound.depth(),
        "transpiled": {
            "depth": tqc.depth(),
            "size": tqc.size(),
            "ops": t_ops,
            "two_qubit_gates": count_two_qubit_gates(tqc),
            "two_qubit_gates_counted_by": "gate arity (basis-independent)",
            "optimization_level": optimization_level,
        },
        "backend": label,
        "noise_model_applied": noise_model is not None,
    }
