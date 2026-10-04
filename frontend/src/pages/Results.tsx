import { useMemo, useState } from "react";
import {
  CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";
import { useRun } from "../lib/store";
import { Disclaimer, NoRun, NotEvaluated, Section, SequenceView, Stat } from "../components/Common";
import { fmt, fmtInt } from "../lib/api";

const AXIS = { fill: "#8b96b8", fontSize: 11 };
const TIP = { background: "#141a2e", border: "1px solid #2a3457", borderRadius: 8 };

type SortKey = "direct_ml_delta_score" | "qubo_energy" | "n_mutations" | "surrogate_error" | "probability";

export default function Results() {
  const { run } = useRun();
  const [sortKey, setSortKey] = useState<SortKey>("direct_ml_delta_score");
  const [onlyImproving, setOnlyImproving] = useState(false);
  const [onlyPareto, setOnlyPareto] = useState(false);

  // All hooks below run unconditionally; the early return follows them.
  const parent = run?.parent_analysis;
  const cands: any[] = run?.candidates ?? [];
  const pareto = run?.pareto ?? {};
  const sv = run?.surrogate_validation ?? {};

  const paretoSeqs = useMemo(
    () => new Set((pareto.pareto_front ?? []).map((c: any) => c.sequence)),
    [pareto]
  );

  const shown = useMemo(() => {
    let list = [...cands];
    if (onlyImproving) list = list.filter((c) => c.direct_ml_delta_score > 0);
    if (onlyPareto) list = list.filter((c) => paretoSeqs.has(c.sequence));
    list.sort((a, b) => {
      const av = a[sortKey] ?? 0, bv = b[sortKey] ?? 0;
      return sortKey === "n_mutations" ? av - bv : bv - av;
    });
    return list;
  }, [cands, sortKey, onlyImproving, onlyPareto, paretoSeqs]);

  const scatter = useMemo(
    () =>
      (pareto.all_points ?? []).map((p: any) => ({
        x: p.activity_norm,
        y: -p.hemolysis_norm,
        seq: p.sequence,
        muts: p.mutations?.join(", ") || "(parent)",
        n: p.n_mutations,
        isPareto: p.is_pareto_optimal,
      })),
    [pareto]
  );

  const best = shown[0];

  if (!parent || !cands.length) return <NoRun />;

  return (
    <div>
      <Section
        title="Parent versus best candidate"
        subtitle="Candidates are ranked by the DIRECT ML score, recomputed by running the property models on the reconstructed sequence — not by QUBO energy. The surrogate's fidelity is measured, so its ranking is not trusted for the final answer."
      >
        <div className="grid md:grid-cols-2 gap-4">
          <div className="card">
            <div className="label mb-2">Parent</div>
            <SequenceView sequence={parent.sequence} />
            <div className="grid grid-cols-2 gap-2 mt-4">
              <Stat label="Predicted activity" value={fmt(parent.model_predicted_activity, 3)} />
              <Stat label="Predicted hemolysis" value={fmt(parent.model_predicted_hemolysis, 3)} />
              <Stat label="Net charge" value={fmt(parent.net_charge_ph74, 2)} />
              <Stat label="Score S(P)" value={fmt(parent.score, 4)} />
            </div>
          </div>
          {best && (
            <div className="card border-accent/40">
              <div className="label mb-2">
                Best candidate by direct ML score
                {paretoSeqs.has(best.sequence) && (
                  <span className="pill bg-accent/15 text-accent ml-2">Pareto-optimal</span>
                )}
              </div>
              <SequenceView sequence={best.sequence} parent={parent.sequence} />
              <div className="flex flex-wrap gap-1 mt-2">
                {best.mutations.map((m: string) => (
                  <span key={m} className="pill bg-accent/15 text-accent mono">{m}</span>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2 mt-4">
                <Stat label="Predicted activity" value={fmt(best.direct_ml_activity, 3)}
                      tone={best.direct_ml_activity > parent.model_predicted_activity ? "good" : "bad"}
                      sub={`parent ${fmt(parent.model_predicted_activity, 3)}`} />
                <Stat label="Predicted hemolysis" value={fmt(best.direct_ml_hemolysis, 3)}
                      tone={best.direct_ml_hemolysis < parent.model_predicted_hemolysis ? "good" : "bad"}
                      sub={`parent ${fmt(parent.model_predicted_hemolysis, 3)}`} />
                <Stat label="ΔS vs parent" value={fmt(best.direct_ml_delta_score, 4)}
                      tone={best.direct_ml_delta_score > 0 ? "good" : "bad"} />
                <Stat label="Similarity to parent" value={`${fmt(100 * best.similarity_to_parent, 1)}%`} />
              </div>
            </div>
          )}
        </div>
      </Section>

      <Section title="Summary">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Stat label="Unique candidates" value={fmtInt(cands.length)} />
          <Stat label="Improve parent score" value={fmtInt(pareto.summary?.n_improving_on_parent_scalar_score)}
                tone="good" sub="ΔS > 0 under the chosen α, β" />
          <Stat label="Pareto-optimal" value={fmtInt(pareto.n_pareto_optimal)} />
          <Stat label="Dominate the parent" value={fmtInt(pareto.n_candidates_dominating_parent)}
                tone="good" sub="better in one objective, no worse in the other" />
        </div>
      </Section>

      <Section
        title="Pareto front"
        subtitle={pareto.dominance_definition ?? "strict dominance over (normalised activity, negated normalised hemolysis), both maximised"}
      >
        {scatter.length ? (
          <div className="card" style={{ height: 420 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 30, right: 20, bottom: 46, left: 10 }}>
                <CartesianGrid stroke="#2a3457" />
                <XAxis type="number" dataKey="x" name="activity (normalised)" tick={AXIS}
                       label={{ value: "normalised predicted activity →  better", fill: "#8b96b8", fontSize: 11, position: "insideBottom", offset: -24 }} />
                <YAxis type="number" dataKey="y" name="−hemolysis (normalised)" tick={AXIS}
                       label={{ value: "− normalised hemolysis →  better", angle: -90, fill: "#8b96b8", fontSize: 11, position: "insideLeft" }} />
                <ZAxis range={[40, 40]} />
                <Tooltip contentStyle={TIP}
                  content={({ active, payload }: any) => {
                    if (!active || !payload?.length) return null;
                    const d = payload[0].payload;
                    return (
                      <div className="bg-panel border border-line rounded-lg p-2 text-xs">
                        <div className="mono break-all max-w-xs">{d.seq}</div>
                        <div className="text-muted mt-1">{d.muts}</div>
                        <div className="mono mt-1">activity {fmt(d.x, 4)} · −hemolysis {fmt(d.y, 4)}</div>
                        <div className="text-muted">{d.n} mutation(s){d.isPareto ? " · Pareto-optimal" : ""}</div>
                      </div>
                    );
                  }} />
                <Legend wrapperStyle={{ fontSize: 11 }} verticalAlign="top" align="right" />
                <Scatter name="candidates" data={scatter.filter((d: any) => !d.isPareto)} fill="#475569" />
                <Scatter name="Pareto front" data={scatter.filter((d: any) => d.isPareto)} fill="#6ee7b7" />
                <Scatter name="parent" fill="#fbbf24"
                  data={[{ x: parent.activity_normalized, y: -parent.hemolysis_normalized,
                           seq: parent.sequence, muts: "(parent)", n: 0, isPareto: false }]} />
              </ScatterChart>
            </ResponsiveContainer>
            <div className="text-xs text-muted mt-2">
              Up and to the right is better in both objectives. The amber point is the parent.
            </div>
          </div>
        ) : (
          <NotEvaluated what="Pareto analysis" reason={pareto.reason} />
        )}
      </Section>

      <Section
        title="Surrogate error on returned candidates"
        subtitle="The gap between what the QUBO believed and what the models actually say. This is the measurement that justifies ranking by direct ML score."
      >
        {sv.on_returned_candidates?.status === "evaluated" ? (
          <>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="MAE" value={fmt(sv.on_returned_candidates.mae, 4)} />
              <Stat label="RMSE" value={fmt(sv.on_returned_candidates.rmse, 4)} />
              <Stat label="Max abs error" value={fmt(sv.on_returned_candidates.max_abs_error, 4)} />
              <Stat label="Spearman ρ" value={fmt(sv.on_returned_candidates.spearman_rho, 3)} />
            </div>
            {sv.verdict && (
              <div className="card mt-3 space-y-2 text-sm">
                <div>
                  <span className="label">fidelity within a fixed mutation count</span>
                  <p className="mt-1">{sv.verdict.fidelity_within_fixed_mutation_count}</p>
                </div>
                <div>
                  <span className="label">end-to-end outcome</span>
                  <p className="mt-1">{sv.verdict.end_to_end_outcome}</p>
                </div>
                <div className="text-xs text-muted pt-2 border-t border-line">
                  {sv.verdict.validity_regime}
                </div>
              </div>
            )}
          </>
        ) : (
          <NotEvaluated what="Surrogate error summary" reason={sv.on_returned_candidates?.reason} />
        )}
      </Section>

      {run?.trustworthiness?.status === "evaluated" && (
        <Section
          title="How much should these predictions be trusted?"
          subtitle="The models are fitted on natural peptides but applied here to point mutants. These two diagnostics say how far each candidate sits from the training data and how much the bootstrap ensemble disagrees about it. See Research → Mutation extrapolation for the measured accuracy of exactly this kind of prediction on real held-out mutant pairs."
        >
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {["activity", "hemolysis"].map((t) => {
              const nn = run.trustworthiness[`${t}_nearest_identity`];
              const sp = run.trustworthiness[`${t}_prediction_spread`];
              return [
                <Stat key={`${t}-nn`}
                      label={`${t}: nearest training seq`}
                      value={nn?.status === "evaluated"
                        ? `${(100 * nn.median).toFixed(1)}%` : "—"}
                      sub={nn?.status === "evaluated"
                        ? `median identity; min ${(100 * nn.min).toFixed(1)}% over ${nn.n_training_sequences} train seqs`
                        : nn?.reason} />,
                <Stat key={`${t}-sp`}
                      label={`${t}: ensemble spread`}
                      value={sp?.status === "evaluated" ? `±${fmt(sp.median_std, 3)}` : "—"}
                      sub={sp?.status === "evaluated"
                        ? `median over ${sp.n_models} bootstrap models` : sp?.reason} />,
              ];
            })}
          </div>
          <div className="card mt-3 text-xs text-muted">
            Ensemble spread is <strong>model</strong> uncertainty — how much resampling the
            training data moves the prediction. It is not an experimental error bar and
            does not bound how wrong the model may be about biology.
          </div>
        </Section>
      )}

      <Section title={`All candidates (${shown.length})`}>
        <div className="flex flex-wrap items-center gap-3 mb-3">
          <label className="text-xs text-muted">sort by</label>
          <select className="input w-auto py-1 text-xs" value={sortKey}
                  onChange={(e) => setSortKey(e.target.value as SortKey)}>
            <option value="direct_ml_delta_score">direct ML ΔS</option>
            <option value="qubo_energy">QUBO energy</option>
            <option value="n_mutations">mutation count</option>
            <option value="surrogate_error">surrogate error</option>
            <option value="probability">QAOA probability</option>
          </select>
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={onlyImproving} onChange={(e) => setOnlyImproving(e.target.checked)} />
            only ΔS &gt; 0
          </label>
          <label className="flex items-center gap-2 text-xs">
            <input type="checkbox" checked={onlyPareto} onChange={(e) => setOnlyPareto(e.target.checked)} />
            only Pareto-optimal
          </label>
        </div>
        <div className="card overflow-auto" style={{ maxHeight: 620 }}>
          <table className="w-full">
            <thead className="sticky top-0 bg-panel"><tr>
              <th className="th">sequence</th><th className="th">mutations</th><th className="th">#</th>
              <th className="th">ML activity</th><th className="th">ML hemolysis</th>
              <th className="th">direct ΔS</th><th className="th">surrogate ΔŜ</th>
              <th className="th">surr. error</th><th className="th">QUBO energy</th>
              <th className="th">similarity</th>
              <th className="th" title="Highest sequence identity to any sequence the activity model was fitted on. High = interpolation, low = extrapolation.">nearest train</th>
              <th className="th" title="Standard deviation across the 10-model bootstrap ensemble. Model uncertainty, not experimental error.">spread</th>
              <th className="th">prob</th><th className="th">source</th>
            </tr></thead>
            <tbody>
              {shown.map((c, i) => (
                <tr key={c.sequence + i} className={`hover:bg-panel2/50 ${paretoSeqs.has(c.sequence) ? "bg-accent/5" : ""}`}>
                  <td className="td mono text-xs max-w-[220px] break-all">{c.sequence}</td>
                  <td className="td">
                    <div className="flex flex-wrap gap-1">
                      {c.mutations.length
                        ? c.mutations.map((m: string) => (
                            <span key={m} className="pill bg-panel2 mono text-[10px]">{m}</span>
                          ))
                        : <span className="text-muted text-xs">parent</span>}
                    </div>
                  </td>
                  <td className="td mono">{c.n_mutations}</td>
                  <td className="td mono">{fmt(c.direct_ml_activity, 3)}</td>
                  <td className="td mono">{fmt(c.direct_ml_hemolysis, 3)}</td>
                  <td className={`td mono ${c.direct_ml_delta_score > 0 ? "text-accent" : "text-bad"}`}>
                    {fmt(c.direct_ml_delta_score, 4)}
                  </td>
                  <td className="td mono text-muted">{fmt(c.surrogate_delta_score, 4)}</td>
                  <td className="td mono text-xs">{fmt(c.surrogate_error, 4)}</td>
                  <td className="td mono">{fmt(c.qubo_energy, 4)}</td>
                  <td className="td mono text-xs">{fmt(100 * c.similarity_to_parent, 1)}%</td>
                  <td className="td mono text-xs">
                    {c.nearest_training_identity?.activity
                      ? `${(100 * c.nearest_training_identity.activity.max_identity).toFixed(1)}%`
                      : "—"}
                  </td>
                  <td className="td mono text-xs">
                    {c.prediction_spread?.activity
                      ? `±${fmt(c.prediction_spread.activity.std, 3)}`
                      : "—"}
                  </td>
                  <td className="td mono text-xs">{c.probability === null ? "—" : fmt(c.probability, 5)}</td>
                  <td className="td text-[10px] text-muted">{c.source}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {pareto.summary?.language_note && (
        <div className="card border-warn/30 bg-warn/5 text-xs text-slate-300 mb-4">
          {pareto.summary.language_note}
        </div>
      )}
      <Disclaimer />
    </div>
  );
}
