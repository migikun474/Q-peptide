import { useMemo, useState } from "react";
import { useRun } from "../lib/store";
import { NoRun, PassFail, Section, Stat } from "../components/Common";
import { fmt, fmtInt } from "../lib/api";

export default function Qubo() {
  const { run } = useRun();
  const [showAll, setShowAll] = useState(false);
  // All hooks run unconditionally; the early return comes after them.
  const q = run?.qubo;
  const ver = run?.verification ?? {};
  const br = q?.build_report ?? {};
  const Q: number[][] | undefined = q?.Q;
  const n = q?.n_vars ?? 0;

  const maxAbs = useMemo(() => {
    if (!Q) return 1;
    let m = 1e-12;
    for (const row of Q) for (const v of row) m = Math.max(m, Math.abs(v));
    return m;
  }, [Q]);

  const cell = (v: number) => {
    const t = Math.abs(v) / maxAbs;
    if (t < 1e-9) return "bg-panel2";
    if (v > 0) return t > 0.5 ? "bg-rose-400/80" : t > 0.15 ? "bg-rose-400/45" : "bg-rose-400/20";
    return t > 0.5 ? "bg-emerald-400/80" : t > 0.15 ? "bg-emerald-400/45" : "bg-emerald-400/20";
  };

  if (!q) return <NoRun />;

  const varLabel = (i: number) => {
    const e = (q.variable_map ?? []).find((v: any) => v.variable_index === i);
    return e ? e.label : `s${i - q.n_mutation_vars}`;
  };

  const linear = Object.entries(q.linear_terms ?? {});
  const quad = Object.entries(q.quadratic_terms ?? {});
  const shownQuad = showAll ? quad : quad.slice(0, 40);

  return (
    <div>
      <Section
        title="Objective and conventions"
        subtitle="Everything on this page is the QUBO that was actually solved. The matrix convention is stated once and tested against an independently written polynomial evaluator."
      >
        <div className="card space-y-3">
          <div className="mono text-sm bg-ink rounded-lg p-4 leading-relaxed overflow-auto">
            <div className="text-accent2">maximise biologically:</div>
            <div>  ΔŜ(x) = Σᵢ Δᵢ xᵢ + Σᵢ&lt;ⱼ Δᵢⱼ xᵢxⱼ</div>
            <div className="text-accent2 mt-2">minimise as a QUBO:</div>
            <div>  E(x) = −ΔŜ(x) + E_budget(x) + E_conflict(x)</div>
            <div className="text-accent2 mt-2">matrix form:</div>
            <div>  E(x) = xᵀQx + c,  Q symmetric</div>
            <div>  Σᵢ Qᵢᵢxᵢ + 2Σᵢ&lt;ⱼ Qᵢⱼxᵢxⱼ + c   ⟹   Qᵢⱼ = cᵢⱼ / 2</div>
          </div>
          <p className="text-xs text-muted">{br.sign_convention}</p>
        </div>
      </Section>

      <Section title="Problem size and penalties">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <Stat label="Total qubits" value={fmtInt(n)} />
          <Stat label="Mutation variables" value={fmtInt(q.n_mutation_vars)} />
          <Stat label="Slack variables" value={fmtInt(q.n_slack_vars)}
                sub={q.n_slack_vars ? `weights ${JSON.stringify(br.slack_weights)}` : "none (equality mode)"} />
          <Stat label="Budget" value={`K = ${q.budget_k}`} sub={q.budget_mode_meaning} />
          <Stat label="Penalty P" value={fmt(q.penalty_budget, 4)} />
          <Stat label="Objective span" value={fmt(br.objective_span?.span, 4)}
                sub="exact max − min of ΔŜ" />
          <Stat label="Triangle bound B" value={fmt(q.objective_bound_B, 4)}
                sub="Σ|Δᵢ| + Σ|Δᵢⱼ| (looser)" />
          <Stat label="Constant c" value={fmt(q.constant, 3)} />
        </div>
        <div className="card">
          <div className="label mb-2">Penalty derivation</div>
          <p className="text-sm text-slate-300">{br.penalty_strategy}</p>
          {q.n_slack_vars > 0 && (
            <p className="text-xs text-muted mt-2">
              Slack bits represent exactly {JSON.stringify(br.slack_representable_values)},
              i.e. the full range [0, {q.budget_k}] with no over-coverage.
            </p>
          )}
        </div>
      </Section>

      <Section
        title="Mathematical verification"
        subtitle="These are executed checks, not claims. The matrix/polynomial test is what catches the off-diagonal factor-of-two error; the penalty check enumerates every assignment and confirms no infeasible state can beat the feasible optimum."
      >
        <div className="card flex flex-wrap gap-3">
          <PassFail ok={ver.qubo_matrix_vs_polynomial?.passed} label="matrix ≡ polynomial" />
          <PassFail ok={ver.penalty_sufficiency?.passed} label="penalties sufficient" />
          <PassFail ok={ver.slack_range?.passed} label="slack range exact" />
          <PassFail ok={ver.ising_mapping?.passed} label="Ising mapping" />
          <PassFail ok={ver.landscape_identities?.passed} label="Δ identities" />
          <PassFail ok={ver.simulator_agreement?.passed} label="simulator ≡ Qiskit" />
        </div>
        <div className="grid md:grid-cols-2 gap-3 mt-3">
          {ver.qubo_matrix_vs_polynomial && (
            <div className="card-tight text-xs">
              <div className="label mb-1">matrix vs polynomial</div>
              <div>samples: {fmtInt(ver.qubo_matrix_vs_polynomial.n_samples)}</div>
              <div>max deviation: {fmt(ver.qubo_matrix_vs_polynomial.max_abs_deviation)}</div>
            </div>
          )}
          {ver.penalty_sufficiency?.status === "evaluated" && (
            <div className="card-tight text-xs">
              <div className="label mb-1">penalty sufficiency</div>
              <div>assignments: {fmtInt(ver.penalty_sufficiency.n_assignments)}</div>
              <div>feasible: {fmtInt(ver.penalty_sufficiency.n_feasible)}</div>
              <div>margin: {fmt(ver.penalty_sufficiency.margin, 4)}</div>
            </div>
          )}
        </div>
      </Section>

      {Q && n <= 28 && (
        <Section title="Q matrix"
                 subtitle="Red raises energy (penalised / unfavourable), green lowers it. Off-diagonal entries are half the polynomial coefficient, by the stated convention.">
          <div className="card overflow-auto">
            <table className="border-collapse">
              <thead>
                <tr>
                  <th className="w-20" />
                  {Array.from({ length: n }, (_, j) => (
                    <th key={j} className="text-[9px] text-muted font-normal px-0.5 align-bottom"
                        style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", height: 54 }}>
                      {varLabel(j)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Q.map((row, i) => (
                  <tr key={i}>
                    <td className="text-[10px] text-muted pr-2 mono whitespace-nowrap">{varLabel(i)}</td>
                    {row.map((v, j) => (
                      <td key={j} className="p-0">
                        <div className={`w-5 h-5 border border-ink ${cell(v)}`}
                             title={`Q[${varLabel(i)}, ${varLabel(j)}] = ${fmt(v, 5)}`} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      <Section title="Coefficients">
        <div className="grid md:grid-cols-2 gap-4">
          <div className="card overflow-auto" style={{ maxHeight: 420 }}>
            <div className="label mb-2">Linear terms ({linear.length})</div>
            <table className="w-full">
              <thead><tr><th className="th">variable</th><th className="th">mutation</th><th className="th">coefficient</th></tr></thead>
              <tbody>
                {linear.map(([k, v]) => (
                  <tr key={k}>
                    <td className="td mono text-muted">x{k}</td>
                    <td className="td mono">{varLabel(Number(k))}</td>
                    <td className="td mono">{fmt(v as number, 4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="card overflow-auto" style={{ maxHeight: 420 }}>
            <div className="flex items-center justify-between mb-2">
              <div className="label">Quadratic terms ({quad.length})</div>
              {quad.length > 40 && (
                <button className="btn-ghost text-xs py-0.5" onClick={() => setShowAll(!showAll)}>
                  {showAll ? "show fewer" : `show all ${quad.length}`}
                </button>
              )}
            </div>
            <table className="w-full">
              <thead><tr><th className="th">pair</th><th className="th">mutations</th><th className="th">coefficient</th></tr></thead>
              <tbody>
                {shownQuad.map(([k, v]) => {
                  const [i, j] = k.split(",").map(Number);
                  const isConflict = Math.abs(Number(v) - q.penalty_conflict) < 1e-6;
                  return (
                    <tr key={k}>
                      <td className="td mono text-muted">x{i}·x{j}</td>
                      <td className="td mono text-xs">{varLabel(i)} · {varLabel(j)}</td>
                      <td className={`td mono ${isConflict ? "text-bad" : ""}`}>
                        {fmt(v as number, 4)}
                        {isConflict && <span className="text-[10px] ml-1">conflict</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </Section>

      <Section title="Variable mapping"
               subtitle="Binary variable ↔ mutation ↔ (position, original residue, new residue). Deterministic and serialisable.">
        <div className="card overflow-auto">
          <table className="w-full">
            <thead><tr>
              <th className="th">variable</th><th className="th">label</th><th className="th">position (1-based)</th>
              <th className="th">original</th><th className="th">new</th>
            </tr></thead>
            <tbody>
              {(q.variable_map ?? []).map((v: any) => (
                <tr key={v.variable_index}>
                  <td className="td mono text-muted">x{v.variable_index}</td>
                  <td className="td mono font-semibold">{v.label}</td>
                  <td className="td mono">{v.position_1based}</td>
                  <td className="td mono">{v.original_aa}</td>
                  <td className="td mono">{v.new_aa}</td>
                </tr>
              ))}
              {(q.slack_variable_indices ?? []).map((idx: number, t: number) => (
                <tr key={idx} className="text-muted">
                  <td className="td mono">x{idx}</td>
                  <td className="td mono">slack bit {t}</td>
                  <td className="td" colSpan={3}>
                    auxiliary; weight {br.slack_weights?.[t]} in Σxᵢ + s = K
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {ver.ising_report && (
        <Section title="Ising Hamiltonian"
                 subtitle="Obtained by substituting xᵢ = (1 − Zᵢ)/2. Because H_C is diagonal, every basis-state energy is checked against the QUBO.">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
            <Stat label="Qubits" value={fmtInt(ver.ising_report.n_qubits)} />
            <Stat label="Pauli terms" value={fmtInt(ver.ising_report.n_terms)} />
            <Stat label="Single-Z terms" value={fmtInt(ver.ising_report.n_single_z_terms)} />
            <Stat label="ZZ terms" value={fmtInt(ver.ising_report.n_zz_terms)} />
            <Stat label="Identity offset" value={fmt(ver.ising_report.identity_offset, 3)} />
            <Stat label="max |hᵢ|" value={fmt(ver.ising_report.max_abs_h, 4)} />
            <Stat label="max |Jᵢⱼ|" value={fmt(ver.ising_report.max_abs_J, 4)} />
            <Stat label="States checked"
                  value={fmtInt(ver.ising_mapping?.n_states_checked)}
                  sub={ver.ising_mapping?.mode} />
          </div>
          <div className="card text-xs text-muted space-y-1">
            <div><span className="text-slate-300">mapping: </span>{ver.ising_report.mapping}</div>
            <div><span className="text-slate-300">qubit ordering: </span>{ver.ising_report.qubit_ordering}</div>
            <div><span className="text-slate-300">offset: </span>{ver.ising_report.note}</div>
            <div><span className="text-slate-300">max deviation: </span>
              {fmt(ver.ising_mapping?.max_abs_deviation)}</div>
          </div>
        </Section>
      )}
    </div>
  );
}
