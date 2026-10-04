"""Q-Peptide FastAPI application.

Every endpoint computes real results from the trained models and the real quantum
simulator. No endpoint returns placeholder or synthetic values: if a stage cannot run it
returns a ``not evaluated`` record or an explicit HTTP error.

Run:  uvicorn backend.app:app --reload --port 8000
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.models.property_models import BiologicalScorer, ScoreWeights
from backend.optimization.hardware import ibm_credentials_available
from backend.optimization.ising import qubo_to_ising
from backend.optimization.landscape import compute_landscape
from backend.optimization.mutations import compatibility_graph, generate_mutation_set
from backend.optimization.pipeline import (
    OptimizationConfig,
    PipelineStageResults,
    _json_default,
    analyse_parent,
    benchmark_metrics,
    environment_record,
    run_pipeline,
)
from backend.optimization.qaoa import run_qaoa_ideal, sample_distribution
from backend.optimization.qubo import BudgetMode, build_qubo
from backend.optimization.solvers import run_all_classical, solve_exact
from backend.schemas.models import (
    BenchmarkRequest,
    HealthResponse,
    MutationGenerateRequest,
    OptimizeRequest,
    QaoaRunRequest,
    QuboBuildRequest,
    SequenceRequest,
)
from backend.utils.peptide import InvalidSequenceError

DISCLAIMER = (
    "Q-Peptide is a computational design tool. All activity and hemolysis values are "
    "MODEL PREDICTIONS from machine-learning models trained on literature-extracted "
    "data. No experimental validation is claimed or implied; experimental validation "
    "would be required before any biological conclusion."
)

from backend.utils.env import load_dotenv

load_dotenv()  # credentials live in .env, not the shell profile

RESULTS_DIR = Path("results/experiments")
RUNS_DIR = RESULTS_DIR / "api_runs"

app = FastAPI(
    title="Q-Peptide API",
    description=(
        "Quantum-assisted constrained optimization for antimicrobial peptide sequence "
        "design. " + DISCLAIMER
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model loading and the run store
# ---------------------------------------------------------------------------
_scorer: BiologicalScorer | None = None
_scorer_error: str | None = None
_lock = threading.Lock()

# run_id -> {"status": ..., "result": ..., "error": ...}
_runs: dict[str, dict] = {}


def get_scorer() -> BiologicalScorer:
    """Load the property models once, lazily."""
    global _scorer, _scorer_error
    with _lock:
        if _scorer is None and _scorer_error is None:
            try:
                _scorer = BiologicalScorer.load()
            except Exception as exc:
                _scorer_error = f"{type(exc).__name__}: {exc}"
    if _scorer is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"property models are not available ({_scorer_error}). Run "
                "`python -m backend.data.curate && python -m backend.models.train` first."
            ),
        )
    return _scorer


def weighted(alpha: float, beta: float) -> BiologicalScorer:
    return get_scorer().with_weights(ScoreWeights(alpha, beta))


def _budget_mode(value: str) -> BudgetMode:
    return BudgetMode.AT_MOST_K if value == "at_most_k" else BudgetMode.EXACTLY_K


def _build(req, scorer):
    """Shared candidate -> landscape -> QUBO construction for several endpoints."""
    mset = generate_mutation_set(
        req.sequence, scorer,
        target_n=req.target_n, max_per_position=req.max_per_position,
    )
    landscape = compute_landscape(mset, scorer)
    if req.budget_k > mset.n:
        raise HTTPException(
            status_code=422,
            detail=(
                f"budget_k={req.budget_k} exceeds the {mset.n} candidate mutations "
                "generated; lower budget_k or raise target_n"
            ),
        )
    problem = build_qubo(
        landscape,
        budget_k=req.budget_k,
        budget_mode=_budget_mode(req.budget_mode),
        penalty_factor=req.penalty_factor,
        penalty_mode=req.penalty_mode,
    )
    return mset, landscape, problem


def _clean(obj):
    """Round-trip through the JSON encoder so numpy types never reach the serialiser."""
    return json.loads(json.dumps(obj, default=_json_default))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Service health, including whether the models and quantum backend are usable."""
    env = environment_record()
    models_loaded = True
    activity_meta = hemolysis_meta = None
    try:
        s = get_scorer()
        activity_meta = {
            **s.activity.metadata,
            "normalization": s.activity.normalization.as_dict(),
        }
        hemolysis_meta = {
            **s.hemolysis.metadata,
            "normalization": s.hemolysis.normalization.as_dict(),
        }
    except HTTPException:
        models_loaded = False

    quantum_ok = True
    try:
        from qiskit_aer import AerSimulator

        AerSimulator()
    except Exception:
        quantum_ok = False

    return HealthResponse(
        status="ok" if models_loaded and quantum_ok else "degraded",
        models_loaded=models_loaded,
        activity_model=activity_meta,
        hemolysis_model=hemolysis_meta,
        quantum_backend_available=quantum_ok,
        ibm_credentials_present=ibm_credentials_available(),
        versions=env["packages"],
        disclaimer=DISCLAIMER,
    )


@app.post("/api/peptide/analyze")
def analyze_peptide(req: SequenceRequest) -> dict:
    """Physicochemical descriptors and model-predicted properties for one peptide."""
    scorer = get_scorer()
    try:
        return _clean({**analyse_parent(req.sequence, scorer), "disclaimer": DISCLAIMER})
    except InvalidSequenceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/mutations/generate")
def generate_mutations(req: MutationGenerateRequest) -> dict:
    """Candidate mutation set, compatibility graph and the full mutation landscape."""
    scorer = weighted(req.alpha, req.beta)
    try:
        mset = generate_mutation_set(
            req.sequence, scorer,
            target_n=req.target_n, max_per_position=req.max_per_position,
            screen=req.screen,
        )
    except (InvalidSequenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    landscape = compute_landscape(mset, scorer)
    return _clean({
        "mutation_set": mset.as_dict(),
        "compatibility_graph": compatibility_graph(mset),
        "landscape": landscape.as_dict(),
        "disclaimer": DISCLAIMER,
    })


@app.post("/api/qubo/build")
def build_qubo_endpoint(req: QuboBuildRequest) -> dict:
    """Build the QUBO and return coefficients, penalties, mapping and Ising summary."""
    scorer = weighted(req.alpha, req.beta)
    try:
        mset, landscape, problem = _build(req, scorer)
    except (InvalidSequenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    hamiltonian, ising_report = qubo_to_ising(problem)
    from backend.optimization.qubo import (
        verify_matrix_matches_polynomial,
        verify_penalty_sufficiency,
        verify_slack_range,
    )

    return _clean({
        "qubo": problem.as_dict(include_matrix=req.include_matrix),
        "ising": ising_report,
        "ising_terms": [
            {"pauli": str(p), "coefficient": float(np.real(c))}
            for p, c in zip(hamiltonian.paulis, hamiltonian.coeffs)
        ][:400],
        "verification": {
            "matrix_vs_polynomial": verify_matrix_matches_polynomial(problem),
            "penalty_sufficiency": verify_penalty_sufficiency(problem),
            "slack_range": verify_slack_range(problem.budget_k),
        },
        "landscape_summary": landscape.report,
        "disclaimer": DISCLAIMER,
    })


@app.post("/api/benchmark/exact")
def benchmark_exact(req: BenchmarkRequest) -> dict:
    """Brute-force enumeration: the ground truth for every other method."""
    scorer = weighted(req.alpha, req.beta)
    try:
        mset, landscape, problem = _build(req, scorer)
        exact = solve_exact(problem)
    except (InvalidSequenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    decoded = [
        problem.decode(np.asarray(x)) for x in exact.optimal_feasible_solutions[:10]
    ]
    return _clean({
        "exact": exact.as_dict(),
        "uniform_random_success_probability": (
            len(exact.optimal_solutions) / exact.n_assignments
        ),
        "optimal_candidates": decoded,
        "disclaimer": DISCLAIMER,
    })


@app.post("/api/benchmark/classical")
def benchmark_classical(req: BenchmarkRequest) -> dict:
    """Simulated annealing, greedy, local search and uniform random sampling."""
    scorer = weighted(req.alpha, req.beta)
    try:
        mset, landscape, problem = _build(req, scorer)
        exact = solve_exact(problem)
    except (InvalidSequenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    classical = run_all_classical(problem, seed=req.seed)
    return _clean({
        "exact_optimal_energy": exact.optimal_energy,
        "classical": {
            name: {
                **res.as_dict(),
                **benchmark_metrics(res.best_energy, exact),
                "decoded": problem.decode(np.asarray(res.best_x)),
            }
            for name, res in classical.items()
        },
        "note": (
            "all classical solvers receive exactly the same QUBO and no biological "
            "information that QAOA does not also have"
        ),
        "disclaimer": DISCLAIMER,
    })


@app.post("/api/qaoa/run")
def qaoa_run(req: QaoaRunRequest) -> dict:
    """Run QAOA at one depth and return parameters, distribution and circuit metrics."""
    scorer = weighted(req.alpha, req.beta)
    try:
        mset, landscape, problem = _build(req, scorer)
        exact = solve_exact(problem)
    except (InvalidSequenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    res = run_qaoa_ideal(
        problem, p=req.p, optimizer=req.optimizer, maxiter=req.maxiter,
        seed=req.seed % (2**31), optimal_solutions=exact.optimal_solutions,
        n_restarts=req.restarts, shots=req.shots,
    )
    uniform = len(exact.optimal_solutions) / exact.n_assignments
    payload = {
        **res.as_dict(max_distribution=256),
        **benchmark_metrics(res.best_sampled_energy, exact),
        "success_probability_vs_uniform": (
            res.success_probability / uniform if uniform > 0 else None
        ),
        "uniform_random_success_probability": uniform,
        "exact_optimal_energy": exact.optimal_energy,
        "decoded_best": problem.decode(np.asarray(res.best_sampled_x)),
    }

    if req.noise:
        from backend.optimization.hardware import build_depolarizing_noise

        model, spec = build_depolarizing_noise()
        noisy = sample_distribution(
            problem, req.p, np.array(res.final_params), shots=req.shots,
            noise_model=model, seed=req.seed % (2**31),
        )
        noisy.pop("distribution", None)
        payload["noisy"] = {
            **noisy,
            **benchmark_metrics(noisy["best_sampled_energy"], exact),
            "noise_spec": spec.as_dict(),
        }

    # the circuit diagram, generated from the actual bound circuit
    from backend.optimization.qaoa import final_circuit

    circuit = final_circuit(problem, req.p, np.array(res.final_params), measure=True)
    payload["circuit_text"] = circuit.draw(output="text", fold=120).single_string()
    payload["disclaimer"] = DISCLAIMER
    return _clean(payload)


@app.post("/api/benchmark/run")
def benchmark_run(req: BenchmarkRequest) -> dict:
    """Exact, classical and QAOA (p=1..3) on one QUBO, side by side."""
    scorer = weighted(req.alpha, req.beta)
    try:
        mset, landscape, problem = _build(req, scorer)
        exact = solve_exact(problem)
    except (InvalidSequenceError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    uniform = len(exact.optimal_solutions) / exact.n_assignments
    classical = run_all_classical(problem, seed=req.seed)
    rows = [
        {
            "method": "exact_enumeration",
            "best_energy": exact.optimal_energy,
            "absolute_gap": 0.0,
            "reached_optimum": True,
            "runtime_seconds": exact.runtime_seconds,
            "n_evaluations": exact.n_assignments,
            "executed": True,
        }
    ]
    for name, res in classical.items():
        rows.append({
            "method": name,
            "best_energy": res.best_energy,
            "mean_energy": res.mean_energy,
            "std_energy": res.std_energy,
            "runtime_seconds": res.runtime_seconds,
            "n_evaluations": res.n_evaluations,
            "executed": True,
            **benchmark_metrics(res.best_energy, exact),
        })
    for p in (1, 2, 3):
        res = run_qaoa_ideal(
            problem, p=p, optimizer="COBYLA", maxiter=300, seed=req.seed % (2**31),
            optimal_solutions=exact.optimal_solutions, n_restarts=3, shots=8192,
        )
        cm = res.circuit_metrics
        rows.append({
            "method": f"qaoa_p{p}",
            "best_energy": res.best_sampled_energy,
            "final_expectation": res.final_expectation,
            "success_probability": res.success_probability,
            "success_probability_vs_uniform": (
                res.success_probability / uniform if uniform > 0 else None
            ),
            "feasible_probability": res.feasible_probability,
            "runtime_seconds": res.runtime_seconds,
            "n_evaluations": res.n_function_evaluations,
            "logical_depth": cm["logical_depth"],
            "transpiled_depth": cm.get("transpiled", {}).get("depth"),
            "logical_two_qubit_gates": cm["logical_two_qubit_gates"],
            "transpiled_two_qubit_gates": cm.get("transpiled", {}).get("two_qubit_gates"),
            "executed": True,
            **benchmark_metrics(res.best_sampled_energy, exact),
        })

    from backend.optimization.hardware import ibm_credentials_available as _creds

    return _clean({
        "n_qubits": problem.n_vars,
        "exact_optimal_energy": exact.optimal_energy,
        "n_optimal_solutions": len(exact.optimal_solutions),
        "uniform_random_success_probability": uniform,
        "rows": rows,
        "not_evaluated": {
            "noisy_qaoa": "run POST /api/qaoa/run with noise=true, or a full optimization",
            "real_qpu": (
                "available: set QISKIT_IBM_TOKEN and request include_hardware"
                if _creds()
                else "no IBM Quantum token in the environment"
            ),
        },
        "disclaimer": DISCLAIMER,
    })


# ---------------------------------------------------------------------------
# Full optimization runs (asynchronous)
# ---------------------------------------------------------------------------
def _execute_run(run_id: str, req: OptimizeRequest) -> None:
    try:
        cfg = OptimizationConfig(
            parent=req.sequence,
            budget_k=req.budget_k,
            budget_mode=_budget_mode(req.budget_mode),
            alpha=req.alpha,
            beta=req.beta,
            target_n=req.target_n,
            max_per_position=req.max_per_position,
            penalty_factor=req.penalty_factor,
            penalty_mode=req.penalty_mode,
            qaoa_depths=tuple(req.qaoa_depths),
            qaoa_optimizers=tuple(req.qaoa_optimizers),
            qaoa_maxiter=req.qaoa_maxiter,
            qaoa_restarts=req.qaoa_restarts,
            shots=req.shots,
            seed=req.seed,
        )
        results = run_pipeline(
            cfg,
            scorer=get_scorer(),
            include_noise=req.include_noise,
            include_hardware=req.include_hardware,
            include_surrogate_validation=req.include_surrogate_validation,
            verbose=False,
        )
        payload = _clean(results.as_dict())
        payload["optimization_id"] = run_id
        payload["completed_at"] = datetime.now(timezone.utc).isoformat()
        payload["disclaimer"] = DISCLAIMER

        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(payload, indent=2))
        _runs[run_id] = {"status": "completed", "result": payload}
    except Exception as exc:
        _runs[run_id] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


@app.post("/api/optimize")
def optimize(req: OptimizeRequest, background: BackgroundTasks) -> dict:
    """Start a full optimization run. Poll GET /api/results/{id} for the outcome."""
    get_scorer()  # fail fast with 503 if the models are missing
    run_id = uuid.uuid4().hex[:16]
    _runs[run_id] = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "request": req.model_dump(),
    }
    background.add_task(_execute_run, run_id, req)
    return {
        "optimization_id": run_id,
        "status": "running",
        "poll": f"/api/results/{run_id}",
        "budget_mode_meaning": _budget_mode(req.budget_mode).description,
    }


def _saved_pipeline_runs() -> list[Path]:
    """Experiment reports that have the shape of a full pipeline run.

    `backend.experiments.final` writes its runs to results/experiments/ rather than to the
    API's own run directory. Treating them as runs means a fresh checkout that has executed
    the experiments shows real results in the UI immediately, instead of an empty app.
    """
    if not RESULTS_DIR.exists():
        return []
    out = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.parent.name == "api_runs":
            continue
        try:
            head = json.loads(f.read_text())
        except Exception:
            continue
        if isinstance(head, dict) and "config" in head and "qubo" in head:
            out.append(f)
    return out


@app.get("/api/results/{optimization_id}")
def get_results(optimization_id: str) -> dict:
    """Fetch a run by id, from memory, from the API run store, or from an experiment report."""
    entry = _runs.get(optimization_id)
    if entry is None:
        path = RUNS_DIR / f"{optimization_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        exp = RESULTS_DIR / f"{Path(optimization_id).name}.json"
        if exp.exists():
            payload = json.loads(exp.read_text())
            if isinstance(payload, dict) and "config" in payload:
                payload.setdefault("optimization_id", optimization_id)
                return payload
        raise HTTPException(status_code=404, detail=f"no run {optimization_id!r}")
    if entry["status"] == "running":
        return {
            "optimization_id": optimization_id,
            "status": "running",
            "started_at": entry.get("started_at"),
        }
    if entry["status"] == "failed":
        raise HTTPException(status_code=500, detail=entry["error"])
    return entry["result"]


@app.get("/api/results")
def list_results(limit: int = 50) -> dict:
    """List runs held in memory and on disk."""
    items = [
        {"optimization_id": k, "status": v["status"],
         "started_at": v.get("started_at")}
        for k, v in _runs.items()
    ]
    known = {i["optimization_id"] for i in items}
    if RUNS_DIR.exists():
        for f in sorted(RUNS_DIR.glob("*.json"), key=lambda p: -p.stat().st_mtime):
            if f.stem not in known:
                items.append({"optimization_id": f.stem, "status": "completed",
                              "started_at": None, "source": "api"})
                known.add(f.stem)
    for f in sorted(_saved_pipeline_runs(), key=lambda p: -p.stat().st_mtime):
        if f.stem not in known:
            items.append({"optimization_id": f.stem, "status": "completed",
                          "started_at": None, "source": "experiment"})
            known.add(f.stem)
    return {"runs": items[:limit], "count": len(items)}


@app.get("/api/experiments/{name}")
def get_experiment(name: str) -> dict:
    """Serve a saved experiment report (e.g. scaling_benchmark, final_experiment)."""
    safe = Path(name).name
    path = RESULTS_DIR / f"{safe}.json"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"experiment {safe!r} not found. Generate it with "
                "`python -m backend.experiments.scaling` or "
                "`python -m backend.experiments.final`."
            ),
        )
    return json.loads(path.read_text())


@app.get("/api/experiments")
def list_experiments() -> dict:
    """Which saved experiment reports exist."""
    if not RESULTS_DIR.exists():
        return {"experiments": []}
    return {
        "experiments": sorted(
            p.stem for p in RESULTS_DIR.glob("*.json")
        )
    }


@app.get("/api/research")
def research_documents() -> dict:
    """Serve the literature review and bibliography to the Research page."""
    out = {}
    for key, rel in (
        ("literature_review", "research/literature_review.md"),
        ("papers", "research/papers.md"),
        ("scientific_audit", "research/scientific_audit.md"),
    ):
        p = Path(rel)
        out[key] = p.read_text() if p.exists() else None
    curation = Path("data/processed/curation_report.json")
    training = RESULTS_DIR / "ml_training_report.json"
    extrapolation = RESULTS_DIR / "mutation_extrapolation_report.json"
    classifier = RESULTS_DIR / "hemolysis_classifier_report.json"
    out["curation_report"] = (
        json.loads(curation.read_text()) if curation.exists() else None
    )
    out["ml_training_report"] = (
        json.loads(training.read_text()) if training.exists() else None
    )
    out["mutation_extrapolation_report"] = (
        json.loads(extrapolation.read_text()) if extrapolation.exists() else None
    )
    out["hemolysis_classifier_report"] = (
        json.loads(classifier.read_text()) if classifier.exists() else None
    )
    return out
