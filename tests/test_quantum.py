"""Ising mapping, exact/classical solvers and QAOA."""
from __future__ import annotations

import numpy as np
import pytest

from backend.optimization.ising import (
    bitstring_to_x,
    diagonal_energies,
    ising_energy_of_bitstring,
    pauli_label,
    qubo_to_ising,
    verify_ising_mapping,
    x_to_bitstring,
)
from backend.optimization.qaoa import (
    build_qaoa_circuit,
    exact_statevector,
    extract_ising_coefficients,
    feasibility_mask,
    run_qaoa_ideal,
    sample_distribution,
    scaled_diagonal,
    verify_simulator_agreement,
)
from backend.optimization.qubo import QuboPolynomial, QuboProblem
from backend.optimization.solvers import (
    run_all_classical,
    solve_exact,
    solve_greedy,
    solve_local_search,
    solve_random_sampling,
    solve_simulated_annealing,
)


# --- bit-order conventions -------------------------------------------------
def test_pauli_label_is_little_endian():
    # qubit 0 is the RIGHTMOST character
    assert pauli_label(3, [0]) == "IIZ"
    assert pauli_label(3, [2]) == "ZII"
    assert pauli_label(3, [0, 2]) == "ZIZ"
    assert pauli_label(2, []) == "II"


def test_pauli_label_rejects_out_of_range():
    with pytest.raises(IndexError):
        pauli_label(3, [3])


def test_bitstring_conversion_roundtrip():
    for x in ([0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1, 1, 0]):
        xv = np.array(x)
        assert np.array_equal(bitstring_to_x(x_to_bitstring(xv), len(x)), xv)


def test_bitstring_to_x_little_endian():
    # "100" -> qubit2=1, qubit1=0, qubit0=0
    assert np.array_equal(bitstring_to_x("100", 3), np.array([0, 0, 1]))
    assert np.array_equal(bitstring_to_x("001", 3), np.array([1, 0, 0]))


def test_bitstring_handles_spaces_and_short_labels():
    assert np.array_equal(bitstring_to_x("01 10", 4), bitstring_to_x("0110", 4))
    assert np.array_equal(bitstring_to_x("1", 3), np.array([1, 0, 0]))


# --- Ising mapping ---------------------------------------------------------
def test_ising_matches_qubo_on_every_basis_state(small_problem):
    H, report = qubo_to_ising(small_problem)
    assert H.num_qubits == small_problem.n_vars
    res = verify_ising_mapping(small_problem, H, tol=1e-8)
    assert res["mode"] == "exhaustive"
    assert res["n_states_checked"] == 1 << small_problem.n_vars
    assert res["passed"], res["failures"]


def test_ising_matches_qubo_for_exactly_k(exactly_k_problem):
    H, _ = qubo_to_ising(exactly_k_problem)
    res = verify_ising_mapping(exactly_k_problem, H, tol=1e-8)
    assert res["passed"], res["failures"]


def test_ising_hand_computed_single_variable():
    """E(x) = 2 x0  ->  H = 1*I - 1*Z0, so E(0)=0 and E(1)=2."""
    poly = QuboPolynomial(n_vars=1)
    poly.add_linear(0, 2.0)
    Q, c = poly.to_matrix()

    class _Stub:
        polynomial = poly
        n_vars = 1

    H, rep = qubo_to_ising(_Stub())  # type: ignore[arg-type]
    assert rep["identity_offset"] == pytest.approx(1.0)
    assert ising_energy_of_bitstring(H, np.array([0])) == pytest.approx(0.0)
    assert ising_energy_of_bitstring(H, np.array([1])) == pytest.approx(2.0)


def test_ising_hand_computed_pair():
    """E(x) = 4 x0 x1 -> check all four basis states."""
    poly = QuboPolynomial(n_vars=2)
    poly.add_quadratic(0, 1, 4.0)

    class _Stub:
        polynomial = poly
        n_vars = 2

    H, _ = qubo_to_ising(_Stub())  # type: ignore[arg-type]
    for x, expected in (
        ([0, 0], 0.0), ([1, 0], 0.0), ([0, 1], 0.0), ([1, 1], 4.0)
    ):
        assert ising_energy_of_bitstring(H, np.array(x)) == pytest.approx(expected)


def test_diagonal_energies_match_qubo(small_problem):
    H, _ = qubo_to_ising(small_problem)
    diag = diagonal_energies(H, small_problem.n_vars)
    assert len(diag) == 1 << small_problem.n_vars
    for code in range(1 << small_problem.n_vars):
        x = np.array([(code >> b) & 1 for b in range(small_problem.n_vars)])
        assert diag[code] == pytest.approx(small_problem.energy(x), abs=1e-8)


def test_ising_rejects_non_diagonal_operator():
    from qiskit.quantum_info import SparsePauliOp

    H = SparsePauliOp(["XI"], coeffs=[1.0])
    with pytest.raises(ValueError, match="diagonal"):
        ising_energy_of_bitstring(H, np.array([0, 0]))


def test_extract_ising_coefficients_roundtrip(small_problem):
    H, _ = qubo_to_ising(small_problem)
    coeffs = extract_ising_coefficients(H)
    assert coeffs.n_qubits == small_problem.n_vars
    # rebuild the diagonal from h/J/offset and compare
    diag = scaled_diagonal(coeffs, 1.0) + coeffs.offset
    reference = diagonal_energies(H, small_problem.n_vars)
    assert np.allclose(diag, reference, atol=1e-8)


# --- exact solver ----------------------------------------------------------
def test_exact_solver_finds_true_minimum(small_problem):
    res = solve_exact(small_problem)
    n = small_problem.n_vars
    assert res.n_assignments == 1 << n

    brute = min(
        small_problem.energy(np.array([(code >> b) & 1 for b in range(n)]))
        for code in range(1 << n)
    )
    assert res.optimal_energy == pytest.approx(brute, abs=1e-9)


def test_exact_solver_reports_all_degenerate_optima(small_problem):
    res = solve_exact(small_problem)
    assert len(res.optimal_solutions) >= 1
    for x in res.optimal_solutions:
        assert small_problem.energy(x) == pytest.approx(res.optimal_energy, abs=1e-9)


def test_exact_solver_degeneracy_count_is_complete(small_problem):
    res = solve_exact(small_problem)
    n = small_problem.n_vars
    manual = sum(
        1
        for code in range(1 << n)
        if abs(
            small_problem.energy(np.array([(code >> b) & 1 for b in range(n)]))
            - res.optimal_energy
        )
        <= 1e-9
    )
    assert manual == len(res.optimal_solutions)


def test_exact_solver_feasible_count(small_problem):
    res = solve_exact(small_problem)
    n = small_problem.n_vars
    manual = sum(
        1
        for code in range(1 << n)
        if small_problem.is_feasible(np.array([(code >> b) & 1 for b in range(n)]))
    )
    assert res.n_feasible == manual


def test_exact_solver_refuses_oversized_problem(small_problem):
    with pytest.raises(ValueError, match="refused"):
        solve_exact(small_problem, max_vars=small_problem.n_vars - 1)


def test_exact_solver_hand_case():
    """A two-variable QUBO with a known optimum."""
    poly = QuboPolynomial(n_vars=2)
    poly.add_linear(0, -1.0)
    poly.add_linear(1, -2.0)
    poly.add_quadratic(0, 1, 5.0)
    Q, c = poly.to_matrix()
    # E(00)=0, E(10)=-1, E(01)=-2, E(11)=2 -> optimum is x=(0,1) at -2
    energies = {
        (0, 0): 0.0, (1, 0): -1.0, (0, 1): -2.0, (1, 1): 2.0,
    }
    for x, e in energies.items():
        assert poly.evaluate(np.array(x)) == pytest.approx(e)
    assert min(energies.values()) == -2.0


# --- classical solvers -----------------------------------------------------
def test_classical_solvers_never_beat_the_exact_optimum(small_problem):
    exact = solve_exact(small_problem)
    for name, res in run_all_classical(small_problem, seed=3).items():
        assert res.best_energy >= exact.optimal_energy - 1e-8, name


def test_classical_solvers_report_required_fields(small_problem):
    for name, res in run_all_classical(small_problem, seed=3).items():
        d = res.as_dict()
        assert d["method"]
        assert d["n_evaluations"] > 0
        assert d["runtime_seconds"] >= 0.0
        assert isinstance(d["feasible"], bool)
        assert len(d["best_x"]) == small_problem.n_vars


def test_stochastic_solvers_report_spread(small_problem):
    sa = solve_simulated_annealing(small_problem, n_seeds=5, n_sweeps=60, seed=0)
    assert len(sa.per_seed_energies) == 5
    assert sa.mean_energy is not None
    assert sa.std_energy is not None
    assert sa.best_energy <= sa.mean_energy + 1e-9


def test_simulated_annealing_reaches_optimum_on_small_instance(small_problem):
    exact = solve_exact(small_problem)
    sa = solve_simulated_annealing(small_problem, n_seeds=10, n_sweeps=400, seed=0)
    assert sa.best_energy == pytest.approx(exact.optimal_energy, abs=1e-6)


def test_greedy_is_deterministic(small_problem):
    a = solve_greedy(small_problem)
    b = solve_greedy(small_problem)
    assert a.best_energy == pytest.approx(b.best_energy)
    assert np.array_equal(a.best_x, b.best_x)


def test_incremental_delta_energy_matches_full_recomputation(small_problem):
    from backend.optimization.solvers import _delta_energy

    rng = np.random.default_rng(0)
    n = small_problem.n_vars
    for _ in range(50):
        x = rng.integers(0, 2, size=n).astype(np.float64)
        i = int(rng.integers(0, n))
        before = small_problem.energy(x)
        predicted = _delta_energy(small_problem.Q, x, i)
        y = x.copy()
        y[i] = 1.0 - y[i]
        actual = small_problem.energy(y) - before
        assert predicted == pytest.approx(actual, abs=1e-9)


def test_random_sampling_baseline_present(small_problem):
    res = solve_random_sampling(small_problem, n_samples=500, n_seeds=2, seed=0)
    assert res.method == "random_sampling"
    assert res.n_evaluations == 1000


def test_local_search_returns_local_minimum(small_problem):
    from backend.optimization.solvers import _delta_energy

    res = solve_local_search(small_problem, n_seeds=5, seed=0)
    x = res.best_x.astype(np.float64)
    for i in range(small_problem.n_vars):
        assert _delta_energy(small_problem.Q, x, i) >= -1e-9


# --- QAOA ------------------------------------------------------------------
def test_qaoa_circuit_qubit_count(small_problem):
    H, _ = qubo_to_ising(small_problem)
    coeffs = extract_ising_coefficients(H)
    qc, gammas, betas = build_qaoa_circuit(coeffs, p=2)
    assert qc.num_qubits == small_problem.n_vars
    assert len(gammas) == 2 and len(betas) == 2
    assert qc.num_parameters == 4


def test_qaoa_circuit_depth_grows_with_p(small_problem):
    H, _ = qubo_to_ising(small_problem)
    coeffs = extract_ising_coefficients(H)
    depths = []
    for p in (1, 2, 3):
        qc, _, _ = build_qaoa_circuit(coeffs, p)
        depths.append(qc.depth())
    assert depths[0] < depths[1] < depths[2]


def test_qaoa_circuit_has_expected_gate_types(small_problem):
    H, _ = qubo_to_ising(small_problem)
    coeffs = extract_ising_coefficients(H)
    qc, _, _ = build_qaoa_circuit(coeffs, p=1)
    ops = dict(qc.count_ops())
    assert ops.get("h", 0) == small_problem.n_vars
    assert ops.get("rx", 0) == small_problem.n_vars
    assert ops.get("rzz", 0) >= 1


@pytest.mark.parametrize("p", [1, 2, 3])
def test_fast_simulator_matches_qiskit_statevector(small_problem, p):
    report = verify_simulator_agreement(small_problem, p=p, seed=p)
    assert report["passed"], report


def test_statevector_is_normalised(small_problem):
    H, _ = qubo_to_ising(small_problem)
    coeffs = extract_ising_coefficients(H)
    rng = np.random.default_rng(0)
    for p in (1, 2):
        theta = rng.uniform(-np.pi, np.pi, size=2 * p)
        psi = exact_statevector(coeffs, p, theta, scale=max(coeffs.max_abs(), 1e-12))
        assert float(np.sum(np.abs(psi) ** 2)) == pytest.approx(1.0, abs=1e-10)


def test_zero_parameters_give_uniform_superposition(small_problem):
    H, _ = qubo_to_ising(small_problem)
    coeffs = extract_ising_coefficients(H)
    n = small_problem.n_vars
    psi = exact_statevector(coeffs, 1, np.zeros(2), scale=1.0)
    probs = np.abs(psi) ** 2
    assert np.allclose(probs, 1.0 / (1 << n), atol=1e-12)


def test_qaoa_expectation_at_zero_params_equals_mean_energy(small_problem):
    """With gamma=beta=0 the state is uniform, so <H> is the mean QUBO energy."""
    H, _ = qubo_to_ising(small_problem)
    coeffs = extract_ising_coefficients(H)
    diag = diagonal_energies(H, small_problem.n_vars)
    psi = exact_statevector(coeffs, 1, np.zeros(2), scale=1.0)
    expectation = float(np.dot(np.abs(psi) ** 2, diag))
    assert expectation == pytest.approx(float(diag.mean()), abs=1e-8)


def test_feasibility_mask_matches_problem_predicate(small_problem):
    mask = feasibility_mask(small_problem, small_problem.n_vars)
    for code in range(1 << small_problem.n_vars):
        x = np.array([(code >> b) & 1 for b in range(small_problem.n_vars)])
        assert bool(mask[code]) == small_problem.is_feasible(x)


def test_qaoa_run_records_everything(small_problem):
    exact = solve_exact(small_problem)
    res = run_qaoa_ideal(
        small_problem, p=1, optimizer="COBYLA", maxiter=60, seed=0,
        optimal_solutions=exact.optimal_solutions, shots=2048,
    )
    d = res.as_dict()
    for key in (
        "p", "optimizer", "initial_params", "final_params", "n_iterations",
        "n_function_evaluations", "final_expectation", "success_probability",
        "circuit_metrics", "convergence", "shots",
    ):
        assert key in d, key
    assert len(res.final_params) == 2
    assert res.n_function_evaluations > 0
    assert 0.0 <= res.success_probability <= 1.0
    assert 0.0 <= res.feasible_probability <= 1.0
    cm = res.circuit_metrics
    assert cm["logical_depth"] > 0
    assert cm["logical_two_qubit_gates"] > 0
    assert "transpiled" in cm


def test_qaoa_probabilities_sum_to_one(small_problem):
    res = run_qaoa_ideal(small_problem, p=1, maxiter=40, seed=0, shots=1024)
    assert sum(res.distribution.values()) == pytest.approx(1.0, abs=1e-8)


def test_qaoa_never_reports_energy_below_exact_optimum(small_problem):
    exact = solve_exact(small_problem)
    for p in (1, 2):
        res = run_qaoa_ideal(
            small_problem, p=p, maxiter=60, seed=1,
            optimal_solutions=exact.optimal_solutions, shots=2048,
        )
        assert res.best_sampled_energy >= exact.optimal_energy - 1e-8
        assert res.final_expectation >= exact.optimal_energy - 1e-8


def test_qaoa_optimisation_improves_on_initial_point(small_problem):
    res = run_qaoa_ideal(small_problem, p=2, maxiter=200, seed=0, n_restarts=2)
    assert res.convergence
    assert res.final_expectation <= max(res.convergence) + 1e-9


def test_qaoa_decoded_candidate_is_consistent(small_problem):
    exact = solve_exact(small_problem)
    res = run_qaoa_ideal(
        small_problem, p=1, maxiter=40, seed=0,
        optimal_solutions=exact.optimal_solutions, shots=2048,
    )
    decoded = small_problem.decode(np.array(res.best_sampled_x))
    assert decoded["n_mutations"] == int(
        np.sum(small_problem.mutation_part(np.array(res.best_sampled_x)))
    )
    if decoded["feasible"]:
        assert decoded["sequence"] is not None
        assert len(decoded["sequence"]) == len(small_problem.landscape.mutation_set.parent)


def test_qaoa_rejects_unknown_optimizer(small_problem):
    with pytest.raises(ValueError, match="optimizer"):
        run_qaoa_ideal(small_problem, p=1, optimizer="NOT_AN_OPTIMIZER")


def test_success_probability_counts_all_degenerate_optima(small_problem):
    """Success probability must be the SUM over the optimum set, not one bitstring."""
    exact = solve_exact(small_problem)
    res = run_qaoa_ideal(
        small_problem, p=1, maxiter=40, seed=0,
        optimal_solutions=exact.optimal_solutions, shots=1024,
    )
    manual = 0.0
    for x in exact.optimal_solutions:
        key = x_to_bitstring(np.asarray(x))
        manual += res.distribution.get(key, 0.0)
    assert res.success_probability == pytest.approx(manual, abs=1e-9)


def test_sampling_reproduces_ideal_distribution(small_problem):
    """Finite sampling of the final circuit should track the exact probabilities."""
    res = run_qaoa_ideal(small_problem, p=1, maxiter=60, seed=0)
    sampled = sample_distribution(
        small_problem, 1, np.array(res.final_params), shots=20000, seed=0
    )
    # compare expectation values; sampling error at 20k shots is small relative to spread
    H, _ = qubo_to_ising(small_problem)
    diag = diagonal_energies(H, small_problem.n_vars)
    spread = float(diag.std())
    assert abs(sampled["expectation"] - res.final_expectation) < 0.1 * spread


def test_sample_distribution_records_transpiled_metrics(small_problem):
    res = run_qaoa_ideal(small_problem, p=1, maxiter=30, seed=0)
    out = sample_distribution(
        small_problem, 1, np.array(res.final_params), shots=1024, seed=0
    )
    assert out["transpiled"]["depth"] > 0
    assert out["transpiled"]["two_qubit_gates"] > 0
    assert out["shots"] == 1024
    assert out["logical_depth"] > 0
    assert sum(out["distribution"].values()) == pytest.approx(1.0, abs=1e-6)
