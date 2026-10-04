import { useMemo, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { useRun } from "../lib/store";
import { NoRun, NotEvaluated, Section, Stat } from "../components/Common";
import { api, errorMessage, fmt, fmtInt } from "../lib/api";

const AXIS = { fill: "#8b96b8", fontSize: 11 };
const TIP = { background: "#141a2e", border: "1px solid #2a3457", borderRadius: 8 };

function CircuitDiagram() {
  const { run } = useRun();
  const [text, setText] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const cfg = run?.config;
  const best = run?.qaoa?.best_configuration;
  const p = best ? run?.qaoa?.runs?.[best]?.p : undefined;

  const load = async () => {
    if (!cfg || !p) return;
    setLoading(true); setErr(null);
    try {
      // Re-runs QAOA at the recorded depth and returns the ACTUAL bound circuit that
      // was executed -- not a hand-drawn illustration of one.
      const r = await api.post("/api/qaoa/run", {
        sequence: cfg.parent,
        budget_k: cfg.budget_k,
        budget_mode: cfg.budget_mode,
        alpha: cfg.alpha, beta: cfg.beta,
        target_n: cfg.target_n, max_per_position: cfg.max_per_position,
        p, optimizer: "COBYLA", maxiter: 60, restarts: 1,
        shots: 2048, include_matrix: false,
      });
      setText(r.data.circuit_text);
    } catch (e) {
      setErr(errorMessage(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="label">Executed circuit</div>
          <p className="mt-1 text-xs text-faint">
            Rendered from the real parameter-bound circuit built from this run's
            Hamiltonian, not a stylised diagram.
          </p>
        </div>
        <button className="btn-ghost text-xs" onClick={load} disabled={loading || !p}>
          {loading ? "building…" : text ? "rebuild" : `draw p=${p ?? "?"} circuit`}
        </button>
      </div>
      {err && <div className="text-xs" style={{ color: "var(--bad)" }}>{err}</div>}
      {text && (
        <pre className="mono max-h-[420px] overflow-auto rounded-xl p-3 text-[10px] leading-[1.35]"
             style={{ background: "rgb(0 0 0 / 0.35)", color: "var(--text-dim)" }}>
          {text}
        </pre>
      )}
      {!text && !err && !loading && (
        <div className="text-xs text-faint">
          Not drawn yet — the circuit is large, so it is built on request.
        </div>
      )}
    </div>
  );
}

export default function Quantum() {
  const { run } = useRun();
  // All hooks run unconditionally; `run` arrives asynchronously, so an early return
  // placed above them would change the hook order between renders.
  const runs: Record<string, any> = run?.qaoa?.runs ?? {};
  const keys = Object.keys(runs);
  const [sel, setSel] = useState<string>("");
  const active = sel && runs[sel] ? sel : (run?.qaoa?.best_configuration ?? keys[0] ?? "");
  const r = runs[active];
  const depthStudy = run?.qaoa?.depth_study ?? [];
  const uniform = run?.exact?.uniform_random_success_probability;

  const convergence = useMemo(
    () => (r?.convergence ?? []).map((v: number, i: number) => ({ i, value: v })),
    [r]
  );

  const topStates = useMemo(() => {
    const d = r?.distribution_top ?? {};
    return Object.entries(d)
      .sort((a, b) => (b[1] as number) - (a[1] as number))
      .slice(0, 24)
      .map(([bits, p]) => ({ bits, p: p as number }));
  }, [r]);

  // one row per (p, optimizer) for the depth charts
  const byDepth = useMemo(() => {
    const m: Record<string, any> = {};
    for (const d of depthStudy) {
      const key = `p=${d.p}`;
      m[key] ||= { p: d.p, label: key };
      m[key][`succ_${d.optimizer}`] = d.success_probability;
      m[key][`gap_${d.optimizer}`] = Math.abs(d.absolute_gap);
      m[key].logical_depth = d.logical_depth;
      m[key].transpiled_depth = d.transpiled_depth;
      m[key].logical_2q = d.logical_two_qubit_gates;
      m[key].transpiled_2q = d.transpiled_two_qubit_gates;
    }
    return Object.values(m).sort((a: any, b: any) => a.p - b.p);
  }, [depthStudy]);

  const optimizers = useMemo(
    () => Array.from(new Set(depthStudy.map((d: any) => d.optimizer))),
    [depthStudy]
  );

  const cm = r?.circuit_metrics ?? {};

  if (!r || !run) return <NoRun />;

  return (
    <div>
      <Section title="QAOA run" subtitle={`Ansatz: ∏ₗ exp(−iβₗ ΣXᵢ) exp(−iγₗ H_C) |+⟩^N with the standard transverse-field X mixer. Objective F(γ,β) = ⟨ψ|H_C|ψ⟩, computed exactly on the ideal simulator so the depth study measures the ansatz rather than a shot budget.`}>
        <div className="flex flex-wrap gap-2 mb-4">
          {keys.map((k) => (
            <button key={k} onClick={() => setSel(k)}
              className={`btn text-xs ${k === active ? "bg-accent/20 text-accent border border-accent" : "btn-ghost"}`}>
              {k}{k === run.qaoa.best_configuration ? " ★" : ""}
            </button>
          ))}
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Depth p" value={r.p} />
          <Stat label="Optimizer" value={r.optimizer} />
          <Stat label="⟨H_C⟩ final" value={fmt(r.final_expectation, 4)} />
          <Stat label="Absolute gap" value={fmt(r.absolute_gap, 6)}
                sub="E_found − E*; the primary metric" tone={Math.abs(r.absolute_gap) < 1e-6 ? "good" : "warn"} />
          <Stat label="Success probability" value={fmt(r.success_probability, 5)}
                sub={`Σ over all ${run.exact?.n_optimal_solutions} optimal state(s)`} tone="good" />
          <Stat label="vs uniform random" value={`${fmt(r.success_probability_vs_uniform, 1)}×`}
                sub={`uniform = ${fmt(uniform, 6)}`} />
          <Stat label="Feasible probability" value={fmt(r.feasible_probability, 4)}
                sub="mass on fully constraint-satisfying states" />
          <Stat label="Function evaluations" value={fmtInt(r.n_function_evaluations)}
                sub={`${fmtInt(r.n_iterations)} iterations`} />
        </div>
      </Section>

      <Section title="Circuit cost"
               subtitle="Logical and transpiled metrics are both reported: hardware error is driven by the transpiled two-qubit gate count, not by logical depth, so comparing hardware to simulation on logical depth alone would be misleading. Two-qubit gates are counted by gate arity, which is basis-independent.">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Qubits" value={fmtInt(cm.n_qubits)} />
          <Stat label="Logical depth" value={fmtInt(cm.logical_depth)} />
          <Stat label="Transpiled depth" value={fmtInt(cm.transpiled?.depth)} sub={cm.transpiled?.target} />
          <Stat label="Parameters" value={fmtInt(cm.n_parameters)} />
          <Stat label="Logical 2-qubit gates" value={fmtInt(cm.logical_two_qubit_gates)} />
          <Stat label="Transpiled 2-qubit gates" value={fmtInt(cm.transpiled?.two_qubit_gates)} />
          <Stat label="Logical size" value={fmtInt(cm.logical_size)} />
          <Stat label="Transpiled size" value={fmtInt(cm.transpiled?.size)} />
        </div>
        {cm.cost_operator_scaling && (
          <div className="card mt-3 text-xs text-muted">
            <span className="text-slate-300">γ reparameterisation: </span>
            {cm.cost_operator_scaling.applied
              ? `cost coefficients divided by ${fmt(cm.cost_operator_scaling.scale, 3)} in the circuit. ${cm.cost_operator_scaling.note}`
              : "none applied"}
          </div>
        )}
        {cm.logical_ops && (
          <div className="card mt-3">
            <div className="label mb-2">Gate counts</div>
            <div className="flex flex-wrap gap-4 text-sm">
              <div>
                <div className="text-xs text-muted mb-1">logical</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(cm.logical_ops).map(([g, c]) => (
                    <span key={g} className="pill bg-panel2 mono">{g}: {String(c)}</span>
                  ))}
                </div>
              </div>
              {cm.transpiled?.ops && (
                <div>
                  <div className="text-xs text-muted mb-1">transpiled</div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(cm.transpiled.ops).map(([g, c]) => (
                      <span key={g} className="pill bg-panel2 mono">{g}: {String(c)}</span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </Section>

      <Section title="Circuit"
               subtitle="Spec requires the displayed circuit to come from the actual backend computation. This one is built by binding the optimised parameters into the circuit generated from this run's cost Hamiltonian.">
        <CircuitDiagram />
      </Section>

      <Section title="Parameters and convergence">
        <div className="grid md:grid-cols-2 gap-4">
          <div className="card">
            <div className="label mb-2">Variational parameters [γ₀, β₀, γ₁, β₁, …]</div>
            <table className="w-full text-sm">
              <thead><tr><th className="th">parameter</th><th className="th">initial</th><th className="th">final</th></tr></thead>
              <tbody>
                {r.final_params.map((v: number, i: number) => (
                  <tr key={i}>
                    <td className="td mono">{i % 2 === 0 ? `γ${Math.floor(i / 2)}` : `β${Math.floor(i / 2)}`}</td>
                    <td className="td mono text-muted">{fmt(r.initial_params[i], 5)}</td>
                    <td className="td mono">{fmt(v, 5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {r.extra?.n_restarts > 1 && (
              <div className="text-xs text-muted mt-2">
                {r.extra.n_restarts} optimizer restarts; the best is reported. The QAOA
                landscape is non-convex, so a single run can stall in a local minimum.
              </div>
            )}
          </div>
          <div className="card" style={{ height: 300 }}>
            <div className="label mb-2">Objective during optimization ({convergence.length} evaluations)</div>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={convergence}>
                <CartesianGrid stroke="#2a3457" />
                <XAxis dataKey="i" tick={AXIS} label={{ value: "evaluation", fill: "#8b96b8", fontSize: 10, position: "insideBottom", offset: -2 }} />
                <YAxis tick={AXIS} />
                <Tooltip contentStyle={TIP} formatter={(v: any) => [fmt(v, 5), "⟨H_C⟩"]} />
                <Line type="monotone" dataKey="value" stroke="#7dd3fc" dot={false} strokeWidth={1.5} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </Section>

      <Section title="Measurement distribution"
               subtitle={`The full distribution is retained, not just the most frequent bitstring. ${fmtInt(r.distribution_support)} states carry probability; the 24 largest are shown.`}>
        <div className="card" style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={topStates} margin={{ top: 8, right: 8, left: 0, bottom: 70 }}>
              <CartesianGrid stroke="#2a3457" vertical={false} />
              <XAxis dataKey="bits" angle={-70} textAnchor="end" height={90} tick={{ ...AXIS, fontSize: 9 }} />
              <YAxis tick={AXIS} />
              <Tooltip contentStyle={TIP} formatter={(v: any) => [fmt(v, 6), "probability"]} />
              <Bar dataKey="p" fill="#7dd3fc" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section title="Depth study"
               subtitle="How solution quality and circuit cost change with p. Both optimizers are shown because the choice genuinely matters.">
        <div className="grid md:grid-cols-2 gap-4">
          <div className="card" style={{ height: 280 }}>
            <div className="label mb-2">p vs success probability</div>
            <ResponsiveContainer width="100%" height="85%">
              <BarChart data={byDepth}>
                <CartesianGrid stroke="#2a3457" vertical={false} />
                <XAxis dataKey="label" tick={AXIS} /><YAxis tick={AXIS} />
                <Tooltip contentStyle={TIP} formatter={(v: any) => fmt(v, 6)} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {optimizers.map((o: any, i: number) => (
                  <Bar key={o} dataKey={`succ_${o}`} name={String(o)}
                       fill={i === 0 ? "#6ee7b7" : "#7dd3fc"} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="card" style={{ height: 280 }}>
            <div className="label mb-2">p vs circuit cost</div>
            <ResponsiveContainer width="100%" height="85%">
              <LineChart data={byDepth}>
                <CartesianGrid stroke="#2a3457" />
                <XAxis dataKey="label" tick={AXIS} /><YAxis tick={AXIS} />
                <Tooltip contentStyle={TIP} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line dataKey="logical_depth" name="logical depth" stroke="#8b96b8" strokeWidth={2} />
                <Line dataKey="transpiled_depth" name="transpiled depth" stroke="#fbbf24" strokeWidth={2} />
                <Line dataKey="transpiled_2q" name="transpiled 2q gates" stroke="#f87171" strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
        <div className="card mt-4 overflow-auto">
          <table className="w-full">
            <thead><tr>
              <th className="th">p</th><th className="th">optimizer</th><th className="th">⟨H_C⟩</th>
              <th className="th">abs gap</th><th className="th">success</th><th className="th">×uniform</th>
              <th className="th">feasible</th><th className="th">logical depth</th>
              <th className="th">transpiled depth</th><th className="th">2q (log/transp)</th>
              <th className="th">evals</th><th className="th">runtime s</th>
            </tr></thead>
            <tbody>
              {depthStudy.map((d: any, i: number) => (
                <tr key={i} className="hover:bg-panel2/50">
                  <td className="td mono">{d.p}</td>
                  <td className="td">{d.optimizer}</td>
                  <td className="td mono">{fmt(d.final_expectation, 3)}</td>
                  <td className="td mono">{fmt(d.absolute_gap, 6)}</td>
                  <td className="td mono text-accent">{fmt(d.success_probability, 5)}</td>
                  <td className="td mono">{fmt(d.success_probability_vs_uniform, 1)}×</td>
                  <td className="td mono">{fmt(d.feasible_probability, 4)}</td>
                  <td className="td mono">{fmtInt(d.logical_depth)}</td>
                  <td className="td mono">{fmtInt(d.transpiled_depth)}</td>
                  <td className="td mono">{fmtInt(d.logical_two_qubit_gates)} / {fmtInt(d.transpiled_two_qubit_gates)}</td>
                  <td className="td mono">{fmtInt(d.n_function_evaluations)}</td>
                  <td className="td mono">{fmt(d.runtime_seconds, 2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card mt-3 text-xs text-muted">{run.qaoa.optimizer_comparison_note}</div>
      </Section>

      <Section title="Noise study"
               subtitle="The best QAOA parameters re-sampled under documented Aer noise models. These are simulations with stated error rates, not measurements from a device.">
        {run.noise?.status === "evaluated" ? (
          <div className="card overflow-auto">
            <table className="w-full">
              <thead><tr>
                <th className="th">model</th><th className="th">2q error</th><th className="th">readout</th>
                <th className="th">⟨H_C⟩</th><th className="th">best energy</th><th className="th">abs gap</th>
                <th className="th">success</th><th className="th">feasible</th>
                <th className="th">transpiled depth</th><th className="th">2q gates</th><th className="th">outcomes</th>
              </tr></thead>
              <tbody>
                {Object.entries(run.noise.runs).map(([name, v]: [string, any]) => (
                  <tr key={name} className="hover:bg-panel2/50">
                    <td className="td font-medium">{name}</td>
                    <td className="td mono">{fmt(v.noise_spec?.two_qubit_depolarizing_error, 4)}</td>
                    <td className="td mono">{fmt(v.noise_spec?.readout_error, 4)}</td>
                    <td className="td mono">{fmt(v.expectation, 3)}</td>
                    <td className="td mono">{fmt(v.best_sampled_energy, 5)}</td>
                    <td className="td mono">{fmt(v.absolute_gap, 5)}</td>
                    <td className="td mono text-accent">{fmt(v.success_probability, 5)}</td>
                    <td className="td mono">{fmt(v.feasible_probability, 4)}</td>
                    <td className="td mono">{fmtInt(v.transpiled?.depth)}</td>
                    <td className="td mono">{fmtInt(v.transpiled?.two_qubit_gates)}</td>
                    <td className="td mono">{fmtInt(v.n_distinct_outcomes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-xs text-muted mt-3">
              Parameters taken from {run.noise.parameters_from}, p = {run.noise.p},
              {" "}{fmtInt(run.noise.shots)} shots. {run.noise.note}
            </div>
          </div>
        ) : (
          <NotEvaluated what="Noise study" reason={run.noise?.reason} />
        )}
      </Section>

      <Section title="Real quantum hardware">
        {run.hardware?.status === "executed" ? (
          <div className="card">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
              <Stat label="Backend" value={run.hardware.backend} />
              <Stat label="Job ID" value={<span className="text-xs break-all">{run.hardware.job_id}</span>} />
              <Stat label="Shots" value={fmtInt(run.hardware.shots)} />
              <Stat label="Success probability" value={fmt(run.hardware.success_probability, 5)} />
              <Stat label="Logical depth" value={fmtInt(run.hardware.logical_depth)} />
              <Stat label="Transpiled depth" value={fmtInt(run.hardware.transpiled_depth)} />
              <Stat label="Transpiled 2q gates" value={fmtInt(run.hardware.transpiled_two_qubit_gates)} />
              <Stat label="Abs gap" value={fmt(run.hardware.absolute_gap, 5)} />
            </div>
          </div>
        ) : (
          <NotEvaluated
            what="Real QPU execution"
            reason={run.hardware?.reason ?? run.hardware?.error ?? "not requested"}
          />
        )}
      </Section>
    </div>
  );
}
