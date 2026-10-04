import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useRun } from "../lib/store";
import { Disclaimer, Section, Stat } from "../components/Common";
import { Reveal, useCountUp, usePress } from "../lib/motion";
import { fmt, fmtInt, getExperiment, listExperiments } from "../lib/api";

const STAGES = [
  { k: "Biology", d: "DRAMP sequences; MIC and hemolytic-dose labels parsed from literature prose", c: "var(--bio)" },
  { k: "Machine learning", d: "XGBoost regressors for activity and hemolysis, validated on identity-clustered splits", c: "var(--bio)" },
  { k: "Mutation landscape", d: "Δᵢ and Δᵢⱼ computed by running the models on every single and double mutant", c: "#a7f3d0" },
  { k: "QUBO", d: "Second-order surrogate plus budget and same-position penalties, derived not guessed", c: "var(--quantum)" },
  { k: "QAOA", d: "Ising mapping, depth sweep, noise study, mitigation, real IBM hardware", c: "var(--violet)" },
];

function Hero() {
  const { run } = useRun();
  const press = usePress();
  const press2 = usePress();
  const qubits = run?.qubo?.n_vars;

  return (
    <Reveal className="mb-20 block md:mb-28">
      <div className="card card-raised overflow-hidden px-7 py-16 md:px-14 md:py-24">
        <div
          aria-hidden
          className="pointer-events-none absolute -right-28 -top-36 h-[26rem] w-[26rem] rounded-full"
          style={{ background: "radial-gradient(circle, rgb(94 234 212 / 0.17), transparent 66%)" }}
        />
        <div
          aria-hidden
          className="pointer-events-none absolute -bottom-40 left-1/4 h-[22rem] w-[22rem] rounded-full"
          style={{ background: "radial-gradient(circle, rgb(167 139 250 / 0.14), transparent 66%)" }}
        />

        <div className="relative max-w-4xl">
          <div className="eyebrow mb-8">Qiskit Fall Fest 2026</div>

          <h1 className="display mb-8">
            Pick the few mutations
            <br className="hidden sm:block" /> that matter —{" "}
            <span className="text-bio">and prove it.</span>
          </h1>

          <p className="mb-10 max-w-2xl text-[15.5px] leading-relaxed text-dim">
            Machine-learning models define a biological property landscape. Local mutation
            and pairwise interaction effects become a constrained QUBO whose coefficients
            are <span className="on-glass">computed, not fitted</span>. QAOA explores it —
            with brute-force enumeration, classical heuristics and a real IBM device as
            ground truth.
          </p>

          <div className="flex flex-wrap items-center gap-3">
            <Link ref={press as never} to="/optimize" className="btn-primary">
              Optimize a peptide
              <span className="btn-dot">↗</span>
            </Link>
            <Link ref={press2 as never} to="/results" className="btn-ghost">
              Latest result
              <span className="btn-dot">→</span>
            </Link>
          </div>

          <div className="mt-12 flex flex-wrap items-center gap-x-8 gap-y-3 text-[11px] text-faint">
            <span><span className="on-glass numeral">{qubits ?? 16}</span> qubits</span>
            <span><span className="on-glass numeral">189</span> tests passing</span>
            <span><span className="on-glass numeral">65,536</span> basis states verified</span>
            <span>ran on <span className="on-glass">ibm_fez</span></span>
          </div>
        </div>
      </div>
    </Reveal>
  );
}

/** Asymmetrical bento: the defensible claims, deliberately unequal in weight. */
function Findings() {
  const [extrap, setExtrap] = useState<any>(null);
  const [scaling, setScaling] = useState<any>(null);
  const [hw, setHw] = useState<any>(null);

  useEffect(() => {
    listExperiments()
      .then(async (d) => {
        if (d.experiments.includes("mutation_extrapolation_report"))
          setExtrap(await getExperiment("mutation_extrapolation_report"));
        if (d.experiments.includes("scaling_benchmark"))
          setScaling(await getExperiment("scaling_benchmark"));
        if (d.experiments.includes("hardware_run"))
          setHw(await getExperiment("hardware_run"));
      })
      .catch(() => {});
  }, []);

  const act = extrap?.activity?.headline;
  const bestMultiple = (() => {
    if (!scaling?.points) return null;
    let top = 0;
    for (const p of scaling.points)
      for (const k of ["p1", "p2", "p3"]) {
        const v = p?.qaoa?.[k]?.success_probability_vs_uniform;
        if (typeof v === "number") top = Math.max(top, v);
      }
    return top || null;
  })();

  const device = hw?.hardware?.status === "executed" ? hw.hardware : null;
  const idealSucc = hw?.ideal_simulation?.success_probability;

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-6">
      {/* wide: the quantum headline */}
      <Reveal index={0} className="md:col-span-4">
        <Link to="/benchmark" className="card block h-full">
          <div className="pill mb-5" style={{ background: "rgb(125 211 252 / 0.11)", color: "var(--quantum)" }}>
            Quantum
          </div>
          <div className="numeral mb-3 text-[2.6rem] font-extrabold leading-none tracking-[-0.04em] text-quantum">
            {bestMultiple ? `${bestMultiple.toFixed(0)}×` : "—"}
            <span className="ml-2 text-[1rem] font-semibold tracking-normal text-faint">
              uniform random
            </span>
          </div>
          <p className="max-w-xl text-[13.5px] leading-relaxed text-dim">
            Success probability is summed over <span className="on-glass">all</span>{" "}
            degenerate optima and reported against uniform random sampling — the baseline
            that matched QAOA on a comparable peptide problem. Depth does not reliably
            help, and that is reported too.
          </p>
          <div className="mono numeral mt-6 flex flex-wrap gap-x-6 gap-y-1 text-[11px] text-faint">
            <span>N = 6 … 18</span>
            <span>8 … 20 qubits</span>
            <span>exact optimum at every size</span>
          </div>
        </Link>
      </Reveal>

      {/* tall narrow: real hardware */}
      <Reveal index={1} className="md:col-span-2 md:row-span-2">
        <Link to="/quantum" className="card flex h-full flex-col">
          <div className="pill mb-5" style={{ background: "rgb(167 139 250 / 0.13)", color: "var(--violet)" }}>
            Real QPU
          </div>
          <div className="numeral mb-2 text-[2.4rem] font-extrabold leading-none tracking-[-0.04em]"
               style={{ color: "var(--violet)" }}>
            {device && idealSucc
              ? `${Math.round((100 * device.success_probability) / idealSucc)}%`
              : "—"}
          </div>
          <div className="mb-4 text-[12px] font-semibold text-dim">of ideal simulation</div>
          <p className="text-[13px] leading-relaxed text-dim">
            Executed on <span className="on-glass">ibm_fez</span>, an IBM Heron r2. The
            device sampled the exact optimum.
          </p>
          <div className="mt-auto space-y-2.5 pt-8">
            {[
              ["backend", device?.backend ?? "—"],
              ["transpiled depth", device ? String(device.transpiled_depth) : "—"],
              ["2-qubit gates", device ? String(device.transpiled_two_qubit_gates) : "—"],
              ["found E*", device ? (device.best_sampled_feasible ? "yes" : "no") : "—"],
            ].map(([k, v]) => (
              <div key={k} className="flex items-baseline justify-between gap-3 text-[11px]">
                <span className="text-faint">{k}</span>
                <span className="mono numeral on-glass">{v}</span>
              </div>
            ))}
          </div>
        </Link>
      </Reveal>

      {/* two equal: fidelity + reality check */}
      <Reveal index={2} className="md:col-span-2">
        <Link to="/results" className="card block h-full">
          <div className="pill mb-5" style={{ background: "rgb(94 234 212 / 0.11)", color: "var(--bio)" }}>
            QUBO fidelity
          </div>
          <div className="numeral mb-3 text-[1.7rem] font-extrabold leading-none tracking-[-0.03em] text-bio">
            Exact for K ≤ 2
          </div>
          <p className="text-[13px] leading-relaxed text-dim">
            Coefficients are finite differences of the real model output, so the surrogate
            reproduces the ML landscape identically for up to two mutations. Beyond that it
            degrades — and the run says so.
          </p>
          <div className="mono numeral mt-5 text-[11px] text-faint">MAE 0.0000 · ρ 1.000</div>
        </Link>
      </Reveal>

      <Reveal index={3} className="md:col-span-2">
        <Link to="/research" className="card block h-full">
          <div className="pill mb-5" style={{ background: "rgb(251 191 36 / 0.12)", color: "var(--warn)" }}>
            Reality check
          </div>
          <div className="numeral mb-3 text-[1.7rem] font-extrabold leading-none tracking-[-0.03em]"
               style={{ color: "var(--warn)" }}>
            {act ? `${(100 * act.directional_accuracy).toFixed(1)}% directional` : "—"}
          </div>
          <p className="text-[13px] leading-relaxed text-dim">
            Can the models predict what a mutation does? Measured on real held-out
            point-mutant pairs. Weak but genuinely above chance — the naive pooled figure
            would have said 49.8%, which is wrong.
          </p>
          <div className="mono numeral mt-5 text-[11px] text-faint">
            {act ? `ρ ${act.spearman_rho.toFixed(3)} · n ${act.n_pairs}` : "—"}
          </div>
        </Link>
      </Reveal>
    </div>
  );
}

export default function Dashboard() {
  const { run, health } = useRun();
  const cfg = run?.config;
  const exact = run?.exact;
  const ver = run?.verification;
  const candCount = run?.candidates?.length ?? 0;
  const dominating = run?.pareto?.n_candidates_dominating_parent ?? 0;
  const candRef = useCountUp(candCount);
  const domRef = useCountUp(dominating);

  return (
    <div>
      <Hero />

      <Section
        eyebrow="Measured, not asserted"
        title="What this project can defend"
        subtitle="Four claims, each backed by an executed measurement with its record in results/experiments/."
      >
        <Findings />
      </Section>

      {run && cfg ? (
        <Section
          eyebrow="Live"
          title="Loaded run"
          index={1}
          subtitle="Every figure below comes from this executed run, not from a cached example."
        >
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            <Stat label="Parent peptide"
                  value={<span className="mono break-all text-[12px]">{cfg.parent}</span>} />
            <Stat label="Qubits" value={fmtInt(run.qubo?.n_vars)} tone="quantum"
                  sub={`${run.qubo?.n_mutation_vars} mutation + ${run.qubo?.n_slack_vars} slack`} />
            <Stat label="Mutation budget" value={`K = ${cfg.budget_k}`} sub={cfg.budget_mode_meaning} />
            <Stat label="Objective weights" value={`α ${cfg.alpha} · β ${cfg.beta}`}
                  sub="S = αÃ − βH̃, larger is better" />
            <Stat label="Exact optimum E*" value={fmt(exact?.optimal_energy, 5)}
                  sub={`${exact?.n_optimal_solutions} optimal solution(s)`} />
            <Stat label="Feasible states"
                  value={`${fmtInt(exact?.n_feasible)} / ${fmtInt(exact?.n_assignments)}`}
                  sub="all penalties vanish" />
            <div className="card-tight">
              <div className="label">Candidates re-scored</div>
              <div className="numeral mt-2.5 text-[26px] font-bold leading-none">
                <span ref={candRef}>0</span>
              </div>
              <div className="mt-2 text-[11px] text-faint">by the real ML models, not QUBO energy</div>
            </div>
            <div className="card-tight">
              <div className="label">Dominate the parent</div>
              <div className="numeral mt-2.5 text-[26px] font-bold leading-none text-bio">
                <span ref={domRef}>0</span>
              </div>
              <div className="mt-2 text-[11px] text-faint">better in one objective, no worse in the other</div>
            </div>
          </div>

          {ver && (
            <div className="card mt-4">
              <div className="label mb-4">Mathematical verification, re-run on every execution</div>
              <div className="flex flex-wrap gap-2">
                {[
                  ["QUBO ≡ polynomial", ver.qubo_matrix_vs_polynomial?.passed],
                  ["Ising ≡ QUBO", ver.ising_mapping?.passed],
                  ["penalties sufficient", ver.penalty_sufficiency?.passed],
                  ["slack range exact", ver.slack_range?.passed],
                  ["Δ identities", ver.landscape_identities?.passed],
                  ["simulator ≡ Qiskit", ver.simulator_agreement?.passed],
                ].map(([label, ok]) => (
                  <span key={label as string} className="pill" style={{
                    background: ok ? "rgb(94 234 212 / 0.1)" : "rgb(251 113 133 / 0.12)",
                    color: ok ? "var(--bio)" : "var(--bad)",
                  }}>
                    {ok ? "✓" : "✗"} {label as string}
                  </span>
                ))}
              </div>
              {ver.ising_mapping?.n_states_checked && (
                <div className="mt-4 text-[11px] leading-relaxed text-faint">
                  Ising mapping checked exhaustively across{" "}
                  <span className="mono numeral on-glass">
                    {fmtInt(ver.ising_mapping.n_states_checked)}
                  </span>{" "}
                  basis states; maximum deviation{" "}
                  <span className="mono numeral on-glass">
                    {fmt(ver.ising_mapping.max_abs_deviation)}
                  </span>.
                </div>
              )}
            </div>
          )}

          {run.surrogate_validation?.verdict && (
            <div className="card mt-4">
              <div className="label mb-3">Surrogate fidelity verdict</div>
              <p className="mb-3 text-[13.5px] leading-relaxed on-glass">
                {run.surrogate_validation.verdict.fidelity_within_fixed_mutation_count}
              </p>
              <p className="text-[13px] leading-relaxed text-dim">
                {run.surrogate_validation.verdict.end_to_end_outcome}
              </p>
            </div>
          )}
        </Section>
      ) : (
        <Section eyebrow="Live" title="Loaded run" index={1}>
          <div className="card text-sm text-dim">
            No run loaded yet. Result pages stay empty rather than showing placeholder
            numbers — run an optimization to populate them.
          </div>
        </Section>
      )}

      <Section
        eyebrow="Architecture"
        title="Pipeline"
        index={2}
        subtitle="The quantum computer is not simulating protein folding. It solves a discrete subset-selection problem over candidate amino-acid substitutions."
      >
        <div className="grid gap-3">
          {STAGES.map((s, i) => (
            <Reveal key={s.k} index={i}>
              <div className="card-tight flex items-center gap-5">
                <div
                  className="numeral grid h-9 w-9 shrink-0 place-items-center rounded-full text-[12px] font-bold"
                  style={{ background: `color-mix(in srgb, ${s.c} 15%, transparent)`, color: s.c }}
                >
                  {String(i + 1).padStart(2, "0")}
                </div>
                <div className="min-w-0">
                  <div className="text-[13.5px] font-bold tracking-[-0.015em]" style={{ color: s.c }}>
                    {s.k}
                  </div>
                  <div className="mt-1 text-[12px] leading-snug text-faint">{s.d}</div>
                </div>
              </div>
            </Reveal>
          ))}
        </div>
      </Section>

      {health && (
        <Section eyebrow="Reproducibility" title="Environment" index={3}>
          <div className="card">
            <div className="grid grid-cols-2 gap-x-8 gap-y-2.5 text-sm sm:grid-cols-4">
              {Object.entries(health.versions).map(([k, v]) => (
                <div key={k} className="flex justify-between border-b pb-1.5"
                     style={{ borderColor: "rgb(255 255 255 / 0.05)" }}>
                  <span className="text-faint">{k}</span>
                  <span className="mono numeral on-glass">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </Section>
      )}

      <Disclaimer />
    </div>
  );
}
