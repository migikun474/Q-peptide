import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from "recharts";
import { useRun } from "../lib/store";
import { NoRun, NotEvaluated, Section, Stat } from "../components/Common";
import { errorMessage, fmt, fmtInt, getExperiment, listExperiments } from "../lib/api";

const AXIS = { fill: "#8b96b8", fontSize: 11 };
const TIP = { background: "#141a2e", border: "1px solid #2a3457", borderRadius: 8 };

export default function Benchmark() {
  const { run } = useRun();
  const [scaling, setScaling] = useState<any>(null);
  const [scalingErr, setScalingErr] = useState<string | null>(null);

  useEffect(() => {
    listExperiments()
      .then((d) => {
        if (d.experiments.includes("scaling_benchmark")) {
          return getExperiment("scaling_benchmark").then(setScaling);
        }
        setScalingErr("scaling_benchmark report has not been generated yet");
      })
      .catch((e) => setScalingErr(errorMessage(e)));
  }, []);

  const rows: any[] = [];
  if (run?.exact) {
    rows.push({
      method: "exact enumeration", group: "exact",
      best_energy: run.exact.optimal_energy, absolute_gap: 0,
      runtime: run.exact.runtime_seconds, evals: run.exact.n_assignments,
      success: null, executed: true,
    });
  }
  for (const [name, v] of Object.entries<any>(run?.classical ?? {})) {
    rows.push({
      method: name.replace(/_/g, " "), group: "classical",
      best_energy: v.best_energy, absolute_gap: v.absolute_gap,
      mean_energy: v.mean_energy, std_energy: v.std_energy,
      runtime: v.runtime_seconds, evals: v.n_evaluations,
      success: null, executed: true,
    });
  }
  for (const d of run?.qaoa?.depth_study ?? []) {
    rows.push({
      method: `QAOA p=${d.p} (${d.optimizer})`, group: "qaoa",
      best_energy: d.final_expectation, absolute_gap: d.absolute_gap,
      runtime: d.runtime_seconds, evals: d.n_function_evaluations,
      success: d.success_probability, executed: true,
      depth: d.logical_depth, tdepth: d.transpiled_depth, t2q: d.transpiled_two_qubit_gates,
    });
  }
  if (run?.noise?.status === "evaluated") {
    for (const [name, v] of Object.entries<any>(run.noise.runs)) {
      if (name === "noiseless") continue;
      rows.push({
        method: `noisy QAOA · ${name}`, group: "noisy",
        best_energy: v.expectation, absolute_gap: v.absolute_gap,
        runtime: v.runtime_seconds, evals: v.shots,
        success: v.success_probability, executed: true,
        tdepth: v.transpiled?.depth, t2q: v.transpiled?.two_qubit_gates,
      });
    }
  }

  const groupColor: Record<string, string> = {
    exact: "bg-slate-500/20 text-slate-300",
    classical: "bg-accent2/15 text-accent2",
    qaoa: "bg-accent/15 text-accent",
    noisy: "bg-warn/15 text-warn",
  };

  const scalePoints = (scaling?.points ?? [])
    .filter((p: any) => p.exact?.optimal_energy !== undefined)
    .map((p: any) => ({
      N: p.n_mutation_vars,
      qubits: p.n_qubits,
      label: `N=${p.n_mutation_vars}`,
      exact_s: p.exact.runtime_seconds,
      sa_gap: Math.abs(p.classical?.simulated_annealing?.absolute_gap ?? 0),
      ls_gap: Math.abs(p.classical?.local_search?.absolute_gap ?? 0),
      greedy_gap: Math.abs(p.classical?.greedy?.absolute_gap ?? 0),
      rand_gap: Math.abs(p.classical?.random_sampling?.absolute_gap ?? 0),
      p1: p.qaoa?.p1?.success_probability,
      p2: p.qaoa?.p2?.success_probability,
      p3: p.qaoa?.p3?.success_probability,
      p3_x: p.qaoa?.p3?.success_probability_vs_uniform,
      uniform: p.exact.uniform_random_success_probability,
      p3_2q: p.qaoa?.p3?.transpiled_two_qubit_gates,
      p3_depth: p.qaoa?.p3?.logical_depth,
    }));

  return (
    <div>
      <Section
        title="Method comparison on one QUBO"
        subtitle="Every method below was actually executed on the identical QUBO. Classical solvers receive no biological information that QAOA does not also have. Methods that were not run are listed as not evaluated rather than given a value."
      >
        {rows.length ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
              <Stat label="Qubits" value={fmtInt(run?.qubo?.n_vars)} />
              <Stat label="Exact optimum E*" value={fmt(run?.exact?.optimal_energy, 6)} />
              <Stat label="Degenerate optima" value={fmtInt(run?.exact?.n_optimal_solutions)} />
              <Stat label="Uniform random success"
                    value={fmt(run?.exact?.uniform_random_success_probability, 6)}
                    sub="one random bitstring being optimal" />
            </div>
            <div className="card overflow-auto">
              <table className="w-full">
                <thead><tr>
                  <th className="th">method</th><th className="th">kind</th>
                  <th className="th">energy</th><th className="th">abs gap</th>
                  <th className="th">mean ± sd</th><th className="th">success</th>
                  <th className="th">evals</th><th className="th">runtime s</th>
                  <th className="th">depth (log/transp)</th><th className="th">2q</th>
                </tr></thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="hover:bg-panel2/50">
                      <td className="td font-medium">{r.method}</td>
                      <td className="td"><span className={`pill ${groupColor[r.group]}`}>{r.group}</span></td>
                      <td className="td mono">{fmt(r.best_energy, 5)}</td>
                      <td className={`td mono ${Math.abs(r.absolute_gap) < 1e-6 ? "text-accent" : ""}`}>
                        {fmt(r.absolute_gap, 6)}
                      </td>
                      <td className="td mono text-xs text-muted">
                        {r.mean_energy !== undefined && r.mean_energy !== null
                          ? `${fmt(r.mean_energy, 4)} ± ${fmt(r.std_energy, 4)}` : "—"}
                      </td>
                      <td className="td mono">{r.success === null || r.success === undefined ? "—" : fmt(r.success, 5)}</td>
                      <td className="td mono">{fmtInt(r.evals)}</td>
                      <td className="td mono">{fmt(r.runtime, 3)}</td>
                      <td className="td mono text-xs">
                        {r.depth ? `${r.depth} / ${r.tdepth ?? "—"}` : r.tdepth ? `— / ${r.tdepth}` : "—"}
                      </td>
                      <td className="td mono">{r.t2q ? fmtInt(r.t2q) : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="card mt-3 text-xs text-muted space-y-1">
              <div>
                <span className="text-slate-300">Why absolute gap: </span>
                QUBO energies are sign-indefinite, so the ratio E_found / E* is meaningless and
                is deliberately not reported.
              </div>
              <div>
                <span className="text-slate-300">Why uniform random is included: </span>
                it is the baseline that matched QAOA on a comparable peptide problem in
                Boulebnane et al. (2022), so it is a required comparison rather than a filler.
              </div>
              <div>
                <span className="text-slate-300">QAOA runtime: </span>
                simulation cost on this workstation, not quantum execution time; it is not
                comparable to the classical solver runtimes as a speed claim.
              </div>
            </div>
            {run?.hardware?.status !== "executed" && (
              <div className="mt-4">
                <NotEvaluated what="Real QPU row"
                              reason={run?.hardware?.reason ?? run?.hardware?.error ?? "not requested"} />
              </div>
            )}
          </>
        ) : (
          <NoRun />
        )}
      </Section>

      <Section
        title="Scaling experiment"
        subtitle="The same pipeline at several problem sizes, each compared against its own exact optimum. These sizes are far too small to support any claim about asymptotic scaling, and no quantum-advantage claim is made."
      >
        {scaling && scalePoints.length ? (
          <>
            <div className="grid md:grid-cols-2 gap-4">
              <div className="card" style={{ height: 300 }}>
                <div className="label mb-2">Problem size vs QAOA success probability</div>
                <ResponsiveContainer width="100%" height="85%">
                  <LineChart data={scalePoints}>
                    <CartesianGrid stroke="#2a3457" />
                    <XAxis dataKey="label" tick={AXIS} />
                    <YAxis tick={AXIS} scale="log" domain={["auto", "auto"]} allowDataOverflow />
                    <Tooltip contentStyle={TIP} formatter={(v: any) => fmt(v, 6)} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Line dataKey="p1" name="QAOA p=1" stroke="#6ee7b7" strokeWidth={2} />
                    <Line dataKey="p2" name="QAOA p=2" stroke="#7dd3fc" strokeWidth={2} />
                    <Line dataKey="p3" name="QAOA p=3" stroke="#a78bfa" strokeWidth={2} />
                    <Line dataKey="uniform" name="uniform random" stroke="#8b96b8"
                          strokeWidth={2} strokeDasharray="4 3" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="card" style={{ height: 300 }}>
                <div className="label mb-2">Problem size vs classical optimality gap</div>
                <ResponsiveContainer width="100%" height="85%">
                  <BarChart data={scalePoints}>
                    <CartesianGrid stroke="#2a3457" vertical={false} />
                    <XAxis dataKey="label" tick={AXIS} /><YAxis tick={AXIS} />
                    <Tooltip contentStyle={TIP} formatter={(v: any) => fmt(v, 6)} />
                    <Legend wrapperStyle={{ fontSize: 11 }} />
                    <Bar dataKey="greedy_gap" name="greedy" fill="#fbbf24" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="ls_gap" name="local search" fill="#f87171" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="sa_gap" name="annealing" fill="#6ee7b7" radius={[3, 3, 0, 0]} />
                    <Bar dataKey="rand_gap" name="random" fill="#8b96b8" radius={[3, 3, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card mt-4 overflow-auto">
              <table className="w-full">
                <thead><tr>
                  <th className="th">N</th><th className="th">qubits</th><th className="th">E*</th>
                  <th className="th">exact s</th><th className="th">feasible</th>
                  <th className="th">SA gap</th><th className="th">greedy gap</th><th className="th">random gap</th>
                  <th className="th">p1 succ</th><th className="th">p2 succ</th><th className="th">p3 succ</th>
                  <th className="th">p3 ×uniform</th><th className="th">p3 depth</th><th className="th">p3 2q</th>
                </tr></thead>
                <tbody>
                  {(scaling.points ?? []).map((p: any, i: number) => {
                    if (p.exact?.status === "not evaluated") {
                      return (
                        <tr key={i}>
                          <td className="td mono">{p.n_mutation_vars}</td>
                          <td className="td mono">{p.n_qubits}</td>
                          <td className="td text-warn text-xs" colSpan={12}>not evaluated — {p.exact.reason}</td>
                        </tr>
                      );
                    }
                    return (
                      <tr key={i} className="hover:bg-panel2/50">
                        <td className="td mono">{p.n_mutation_vars}</td>
                        <td className="td mono">{p.n_qubits}</td>
                        <td className="td mono">{fmt(p.exact.optimal_energy, 5)}</td>
                        <td className="td mono">{fmt(p.exact.runtime_seconds, 3)}</td>
                        <td className="td mono text-xs">{fmtInt(p.exact.n_feasible)}/{fmtInt(p.exact.n_assignments)}</td>
                        <td className="td mono">{fmt(p.classical?.simulated_annealing?.absolute_gap, 5)}</td>
                        <td className="td mono">{fmt(p.classical?.greedy?.absolute_gap, 5)}</td>
                        <td className="td mono">{fmt(p.classical?.random_sampling?.absolute_gap, 5)}</td>
                        <td className="td mono">{fmt(p.qaoa?.p1?.success_probability, 5)}</td>
                        <td className="td mono">{fmt(p.qaoa?.p2?.success_probability, 5)}</td>
                        <td className="td mono text-accent">{fmt(p.qaoa?.p3?.success_probability, 5)}</td>
                        <td className="td mono">{fmt(p.qaoa?.p3?.success_probability_vs_uniform, 1)}×</td>
                        <td className="td mono">{fmtInt(p.qaoa?.p3?.logical_depth)}</td>
                        <td className="td mono">{fmtInt(p.qaoa?.p3?.transpiled_two_qubit_gates)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {scaling.caveats && (
              <div className="card mt-3">
                <div className="label mb-2">Caveats recorded with this experiment</div>
                <ul className="text-xs text-muted space-y-1 list-disc pl-4">
                  {scaling.caveats.map((c: string, i: number) => <li key={i}>{c}</li>)}
                </ul>
              </div>
            )}
          </>
        ) : (
          <NotEvaluated
            what="Scaling benchmark across N = 6 … 18"
            reason={scalingErr ?? "generate it with: python -m backend.experiments.scaling"}
          />
        )}
      </Section>
    </div>
  );
}
