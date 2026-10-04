import { useMemo } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import { useRun } from "../lib/store";
import { NoRun, Section, Stat } from "../components/Common";
import { fmt, fmtInt } from "../lib/api";

export default function Landscape() {
  const { run } = useRun();
  // Every hook runs before the early return: React requires a stable hook order, and
  // `run` goes from null to loaded while this component is mounted.
  const ls = run?.landscape;
  const graph = run?.compatibility_graph;
  const rep = ls?.report ?? {};

  const singles = useMemo(
    () => (ls?.individual_effects ?? []).map((e: any) => ({ ...e, delta: e.delta_i })),
    [ls]
  );
  const pairs = ls?.pairwise_effects ?? [];
  const n = ls?.n_variables ?? singles.length;

  // Δij matrix for the heatmap
  const matrix = useMemo(() => {
    const m: (number | null)[][] = Array.from({ length: n }, () => Array(n).fill(null));
    for (const p of pairs) { m[p.i][p.j] = p.delta_ij; m[p.j][p.i] = p.delta_ij; }
    return m;
  }, [pairs, n]);

  const maxAbs = useMemo(
    () => Math.max(1e-9, ...pairs.map((p: any) => Math.abs(p.delta_ij))),
    [pairs]
  );
  const conflicts = useMemo(() => {
    const s = new Set<string>();
    for (const e of graph?.edges ?? []) if (e.relation === "conflict") s.add(`${e.source}-${e.target}`);
    return s;
  }, [graph]);

  const labels = singles.map((s: any) => s.label);

  if (!ls) return <NoRun />;

  const cellColor = (v: number | null, i: number, j: number) => {
    if (i === j) return "bg-line";
    if (conflicts.has(`${Math.min(i, j)}-${Math.max(i, j)}`)) return "bg-bad/40";
    if (v === null) return "bg-panel2";
    const t = Math.abs(v) / maxAbs;
    if (v > 0) return t > 0.6 ? "bg-emerald-400/80" : t > 0.3 ? "bg-emerald-400/50" : "bg-emerald-400/25";
    return t > 0.6 ? "bg-sky-400/80" : t > 0.3 ? "bg-sky-400/50" : "bg-sky-400/25";
  };

  return (
    <div>
      <Section
        title="Individual mutation effects Δᵢ"
        subtitle="Δᵢ = S(P ⊕ mᵢ) − S(P), obtained by running the property models on each single mutant. No heuristic coefficients are used anywhere."
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <Stat label="Variables N" value={fmtInt(n)} />
          <Stat label="Model evaluations" value={fmtInt(ls.n_model_evaluations)}
                sub="1 + N + compatible pairs" />
          <Stat label="Beneficial Δᵢ > 0" value={fmtInt(rep.delta_i?.n_beneficial)} tone="good" />
          <Stat label="Deleterious Δᵢ < 0" value={fmtInt(rep.delta_i?.n_deleterious)} tone="bad" />
        </div>
        <div className="card" style={{ height: 320 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={singles} margin={{ top: 8, right: 8, left: 0, bottom: 40 }}>
              <CartesianGrid stroke="#2a3457" vertical={false} />
              <XAxis dataKey="label" angle={-45} textAnchor="end" height={60}
                     tick={{ fill: "#8b96b8", fontSize: 11 }} />
              <YAxis tick={{ fill: "#8b96b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ background: "#141a2e", border: "1px solid #2a3457", borderRadius: 8 }}
                formatter={(v: any) => [fmt(v, 5), "Δᵢ"]}
              />
              <Bar dataKey="delta" radius={[3, 3, 0, 0]}>
                {singles.map((s: any, i: number) => (
                  <Cell key={i} fill={s.delta >= 0 ? "#6ee7b7" : "#f87171"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </Section>

      <Section
        title="Pairwise interaction effects Δᵢⱼ"
        subtitle="Δᵢⱼ = S(P ⊕ mᵢ ⊕ mⱼ) − S(P ⊕ mᵢ) − S(P ⊕ mⱼ) + S(P). Zero means additive, positive synergistic, negative antagonistic. This term is why the problem is genuinely quadratic rather than a simple selection."
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <Stat label="Defined pairs" value={fmtInt(rep.delta_ij?.n_defined)} />
          <Stat label="Synergistic" value={fmtInt(rep.delta_ij?.n_synergistic)} tone="good" />
          <Stat label="Antagonistic" value={fmtInt(rep.delta_ij?.n_antagonistic)} tone="bad" />
          <Stat label="|Δᵢⱼ| / |Δᵢ| ratio" value={fmt(rep.interaction_strength_ratio, 3)}
                sub="near 0 = additive; larger = quadratic term matters" />
        </div>

        {rep.interpretation && (
          <div className="card text-xs text-muted mb-4">{rep.interpretation}</div>
        )}

        <div className="card overflow-auto">
          <div className="flex items-center gap-4 mb-3">
            <div className="label">Δᵢⱼ matrix</div>
            <div className="flex items-center gap-3 text-xs">
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-emerald-400/70 inline-block" /> synergy</span>
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-sky-400/70 inline-block" /> antagonism</span>
              <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-bad/40 inline-block" /> conflict (undefined)</span>
            </div>
          </div>
          <table className="border-collapse">
            <thead>
              <tr>
                <th className="w-16" />
                {labels.map((l: string) => (
                  <th key={l} className="text-[9px] text-muted font-normal px-0.5 align-bottom"
                      style={{ writingMode: "vertical-rl", transform: "rotate(180deg)", height: 56 }}>
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {labels.map((rowLabel: string, i: number) => (
                <tr key={rowLabel}>
                  <td className="text-[10px] text-muted pr-2 mono whitespace-nowrap">{rowLabel}</td>
                  {labels.map((_: string, j: number) => (
                    <td key={j} className="p-0">
                      <div
                        className={`w-5 h-5 border border-ink ${cellColor(matrix[i][j], i, j)}`}
                        title={
                          i === j ? `${rowLabel} (diagonal)` :
                          conflicts.has(`${Math.min(i, j)}-${Math.max(i, j)}`)
                            ? `${labels[i]} / ${labels[j]}: same-position conflict, Δᵢⱼ undefined`
                            : `${labels[i]} / ${labels[j]}: Δᵢⱼ = ${fmt(matrix[i][j], 5)}`
                        }
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section
        title="Compatibility graph"
        subtitle="Two substitutions at the same residue position cannot both be selected. Conflicts are held as explicit data and handed to the QUBO builder; the optimizer never parses sequence strings to decide compatibility."
      >
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <Stat label="Nodes" value={fmtInt(graph?.nodes?.length)} />
          <Stat label="Conflict edges" value={fmtInt(graph?.n_conflict_edges)} tone="bad" />
          <Stat label="Compatible edges" value={fmtInt(graph?.n_compatible_edges)} tone="good" />
          <Stat label="Positions used" value={fmtInt(run.mutation_set?.generation_report?.n_positions_used)} />
        </div>
        <div className="card">
          <div className="label mb-3">Mutations grouped by residue position — mutations in the same group are mutually exclusive</div>
          <div className="flex flex-wrap gap-3">
            {(Object.entries(
              (graph?.nodes ?? []).reduce((acc: Record<string, any[]>, nd: any) => {
                const key = String(nd.position_1based);
                (acc[key] ||= []).push(nd);
                return acc;
              }, {} as Record<string, any[]>)
            ) as [string, any[]][])
              .sort((a, b) => Number(a[0]) - Number(b[0]))
              .map(([pos, nodes]) => (
                <div key={pos}
                     className={`rounded-lg border p-2 ${nodes.length > 1 ? "border-bad/50 bg-bad/5" : "border-line"}`}>
                  <div className="text-[10px] text-muted mb-1">
                    position {pos} ({nodes[0].original_aa})
                    {nodes.length > 1 && <span className="text-bad"> · exclusive</span>}
                  </div>
                  <div className="flex gap-1">
                    {nodes.map((nd: any) => {
                      const d = singles.find((s: any) => s.index === nd.id);
                      return (
                        <span key={nd.id}
                              title={`${nd.label}: Δᵢ = ${fmt(d?.delta, 5)}`}
                              className={`pill mono ${d?.delta >= 0 ? "bg-accent/15 text-accent" : "bg-bad/15 text-bad"}`}>
                          {nd.label}
                        </span>
                      );
                    })}
                  </div>
                </div>
              ))}
          </div>
        </div>
      </Section>

      <Section title="Candidate mutation table">
        <div className="card overflow-auto">
          <table className="w-full">
            <thead>
              <tr>
                <th className="th">var</th><th className="th">mutation</th>
                <th className="th">position</th><th className="th">from → to</th>
                <th className="th">Δᵢ</th><th className="th">effect</th>
                <th className="th">conflicts with</th>
              </tr>
            </thead>
            <tbody>
              {singles.map((s: any) => {
                const node = (graph?.nodes ?? []).find((nd: any) => nd.id === s.index);
                const confl = (graph?.edges ?? [])
                  .filter((e: any) => e.relation === "conflict" && (e.source === s.index || e.target === s.index))
                  .map((e: any) => labels[e.source === s.index ? e.target : e.source]);
                return (
                  <tr key={s.index} className="hover:bg-panel2/50">
                    <td className="td mono text-muted">x{s.index}</td>
                    <td className="td mono font-semibold">{s.label}</td>
                    <td className="td mono">{node?.position_1based}</td>
                    <td className="td mono">{node?.original_aa} → {node?.new_aa}</td>
                    <td className={`td mono ${s.delta >= 0 ? "text-accent" : "text-bad"}`}>{fmt(s.delta, 5)}</td>
                    <td className="td">
                      <span className={`pill ${s.beneficial ? "bg-accent/15 text-accent" : "bg-bad/15 text-bad"}`}>
                        {s.beneficial ? "beneficial" : "deleterious"}
                      </span>
                    </td>
                    <td className="td mono text-xs text-muted">{confl.length ? confl.join(", ") : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Section>

      {run.mutation_set?.generation_report?.filters && (
        <Section title="Candidate generation filters">
          <div className="card text-sm space-y-2">
            <div><span className="text-muted">Protected parent residues: </span>
              <span className="mono">{run.mutation_set.generation_report.filters.protected_parent_residues}</span></div>
            <div><span className="text-muted">Forbidden new residues: </span>
              <span className="mono">{run.mutation_set.generation_report.filters.forbidden_new_residues}</span></div>
            <div><span className="text-muted">Minimum hydropathy change: </span>
              <span className="mono">{run.mutation_set.generation_report.filters.min_hydropathy_delta}</span></div>
            <p className="text-xs text-muted pt-2 border-t border-line">
              {run.mutation_set.generation_report.filters.rationale}
            </p>
            <p className="text-xs text-muted">
              {run.mutation_set.generation_report.n_admissible_after_filters} substitutions survived
              filtering and were all scored with the real models; {n} were kept.
              {" "}{run.mutation_set.generation_report.size_justification}
            </p>
          </div>
        </Section>
      )}
    </div>
  );
}
