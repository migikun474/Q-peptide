"""QUBO -> Ising cost Hamiltonian, with exhaustive numerical verification.

Substituting ``x_i = (1 - Z_i) / 2`` into

    E(x) = const + sum_i c_i x_i + sum_{i<j} c_ij x_i x_j

and collecting terms:

    sum_i c_i x_i          = (1/2) sum_i c_i
                             - (1/2) sum_i c_i Z_i

    sum_{i<j} c_ij x_i x_j = (1/4) sum_{i<j} c_ij
                             - (1/4) sum_{i<j} c_ij (Z_i + Z_j)
                             + (1/4) sum_{i<j} c_ij Z_i Z_j

giving

    H_C = offset * I + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    offset = const + (1/2) sum_i c_i + (1/4) sum_{i<j} c_ij
    h_i    = -(1/2) c_i - (1/4) sum_{j != i} c_ij
    J_ij   =  (1/4) c_ij

The identity offset is included so that ``<x|H_C|x> == E_QUBO(x)`` exactly, which makes
the mapping exhaustively testable. It is a constant and so has no effect on which state
minimises the energy, but keeping it means expectation values are directly comparable to
QUBO energies with no bookkeeping.

QUBIT ORDERING. Qiskit Pauli labels are little-endian: the RIGHTMOST character of the
label is qubit 0. Measurement bitstrings follow the same convention, so bit ``i`` of a
counts key ``s`` is ``s[len(s) - 1 - i]``. Both directions go through
:func:`pauli_label` and :func:`bitstring_to_x` so the convention lives in one place.
"""
from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from backend.optimization.qubo import QuboProblem


def pauli_label(n_qubits: int, z_positions: list[int]) -> str:
    """A Pauli label with Z on the given qubit indices, I elsewhere (little-endian)."""
    chars = ["I"] * n_qubits
    for q in z_positions:
        if not 0 <= q < n_qubits:
            raise IndexError(f"qubit {q} out of range for {n_qubits} qubits")
        chars[n_qubits - 1 - q] = "Z"
    return "".join(chars)


def bitstring_to_x(bitstring: str, n_vars: int) -> np.ndarray:
    """Convert a Qiskit measurement bitstring to a binary vector indexed by qubit.

    Handles labels that include spaces (from multiple classical registers) and that may
    be shorter than ``n_vars`` if leading zeros were trimmed.
    """
    s = bitstring.replace(" ", "")
    if len(s) < n_vars:
        s = s.rjust(n_vars, "0")
    x = np.zeros(n_vars, dtype=int)
    for i in range(n_vars):
        x[i] = 1 if s[len(s) - 1 - i] == "1" else 0
    return x


def x_to_bitstring(x: np.ndarray) -> str:
    """Inverse of :func:`bitstring_to_x` (little-endian, qubit 0 rightmost)."""
    return "".join("1" if v else "0" for v in reversed(np.asarray(x).astype(int)))


def qubo_to_ising(problem: QuboProblem) -> tuple[SparsePauliOp, dict]:
    """Build the diagonal cost Hamiltonian ``H_C`` for a QUBO.

    Returns the operator and a report recording the coefficient structure.
    """
    n = problem.n_vars
    poly = problem.polynomial

    c_lin = poly.linear
    c_quad = poly.quadratic

    offset = poly.constant + 0.5 * sum(c_lin.values()) + 0.25 * sum(c_quad.values())

    h = np.zeros(n, dtype=np.float64)
    for i, c in c_lin.items():
        h[i] -= 0.5 * c
    for (i, j), c in c_quad.items():
        h[i] -= 0.25 * c
        h[j] -= 0.25 * c

    labels: list[str] = []
    coeffs: list[float] = []

    if abs(offset) > 1e-15:
        labels.append("I" * n)
        coeffs.append(offset)
    for i in range(n):
        if abs(h[i]) > 1e-15:
            labels.append(pauli_label(n, [i]))
            coeffs.append(float(h[i]))
    for (i, j), c in sorted(c_quad.items()):
        J = 0.25 * c
        if abs(J) > 1e-15:
            labels.append(pauli_label(n, [i, j]))
            coeffs.append(float(J))

    if not labels:  # degenerate all-zero Hamiltonian
        labels, coeffs = ["I" * n], [0.0]

    hamiltonian = SparsePauliOp(labels, coeffs=np.array(coeffs, dtype=complex))

    report = {
        "n_qubits": n,
        "n_terms": len(labels),
        "n_single_z_terms": int(np.sum(np.abs(h) > 1e-15)),
        "n_zz_terms": sum(1 for (_, _), c in c_quad.items() if abs(0.25 * c) > 1e-15),
        "identity_offset": float(offset),
        "mapping": "x_i = (1 - Z_i)/2",
        "qubit_ordering": (
            "Qiskit little-endian: rightmost Pauli character and rightmost measurement "
            "bit are qubit 0"
        ),
        "includes_identity_offset": True,
        "note": (
            "the identity term makes <x|H_C|x> equal E_QUBO(x) exactly; it does not "
            "change the argmin"
        ),
        "max_abs_h": float(np.max(np.abs(h))) if n else 0.0,
        "max_abs_J": (
            float(max(abs(0.25 * c) for c in c_quad.values())) if c_quad else 0.0
        ),
    }
    return hamiltonian, report


def ising_energy_of_bitstring(hamiltonian: SparsePauliOp, x: np.ndarray) -> float:
    """``<x|H_C|x>`` evaluated analytically, without building a statevector.

    For a diagonal Hamiltonian each Pauli term contributes its coefficient times the
    product of ``z_i = 1 - 2 x_i`` over the qubits carrying a Z.
    """
    x = np.asarray(x, dtype=int)
    n = hamiltonian.num_qubits
    z = 1 - 2 * x
    total = 0.0
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = str(pauli)
        prod = 1.0
        for pos, ch in enumerate(label):
            if ch == "Z":
                qubit = n - 1 - pos
                prod *= z[qubit]
            elif ch != "I":
                raise ValueError(
                    f"cost Hamiltonian must be diagonal (I/Z only), found {ch!r}"
                )
        total += float(np.real(coeff)) * prod
    return total


def verify_ising_mapping(
    problem: QuboProblem,
    hamiltonian: SparsePauliOp,
    max_enumerate_vars: int = 20,
    tol: float = 1e-8,
    n_random: int = 4000,
    seed: int = 0,
) -> dict:
    """Assert ``<x|H_C|x> == E_QUBO(x)`` for every basis state (or a random sample).

    Exhaustive for ``n_vars <= max_enumerate_vars``; otherwise a large random sample,
    with the report stating which was done.
    """
    n = problem.n_vars
    exhaustive = n <= max_enumerate_vars

    if exhaustive:
        states = (
            np.array([(code >> b) & 1 for b in range(n)], dtype=int)
            for code in range(1 << n)
        )
        n_checked = 1 << n
    else:
        rng = np.random.default_rng(seed)
        states = (rng.integers(0, 2, size=n) for _ in range(n_random))
        n_checked = n_random

    max_dev = 0.0
    failures = []
    for x in states:
        e_qubo = problem.energy(x)
        e_ising = ising_energy_of_bitstring(hamiltonian, x)
        dev = abs(e_qubo - e_ising)
        if dev > max_dev:
            max_dev = dev
        if dev > tol and len(failures) < 10:
            failures.append(
                {"x": np.asarray(x).tolist(), "qubo": e_qubo, "ising": e_ising}
            )

    return {
        "mode": "exhaustive" if exhaustive else "random_sample",
        "n_states_checked": n_checked,
        "tolerance": tol,
        "max_abs_deviation": max_dev,
        "passed": max_dev <= tol,
        "failures": failures,
    }


def diagonal_energies(hamiltonian: SparsePauliOp, n: int) -> np.ndarray:
    """All ``2^n`` diagonal energies of ``H_C``, vectorised.

    Used to compute exact QAOA expectation values and success probabilities from a
    statevector without re-evaluating the QUBO per state.
    """
    codes = np.arange(1 << n, dtype=np.uint64)
    bits = ((codes[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(
        np.int8
    )
    z = 1 - 2 * bits.astype(np.int64)  # (2^n, n), column i is qubit i

    total = np.zeros(1 << n, dtype=np.float64)
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        label = str(pauli)
        prod = np.ones(1 << n, dtype=np.int64)
        for pos, ch in enumerate(label):
            if ch == "Z":
                prod = prod * z[:, n - 1 - pos]
        total += float(np.real(coeff)) * prod
    return total
