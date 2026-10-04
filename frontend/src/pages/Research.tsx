import { useEffect, useState } from "react";
import type { Dict } from "../lib/api";
import { errorMessage, fmt, fmtInt, getResearch } from "../lib/api";
import { NotEvaluated, Section, Stat } from "../components/Common";
import { MathBlock, MathInline, splitInlineMath } from "../components/Math";

/** Minimal markdown rendering: headings, tables, lists, code, emphasis.
 *  Deliberately small — the documents are ours, so no sanitiser-grade parser is needed,
 *  and text is inserted as text rather than HTML. */
function Markdown({ text }: { text: string }) {
  const lines = text.split("\n");
  const out: React.ReactNode[] = [];
  let i = 0;
  let key = 0;

  // Emphasis, code and links. Math is peeled off first by `inline` below, because
  // LaTeX routinely contains * and _ that would otherwise be eaten as Markdown.
  //
  // Code spans are matched first and never recursed into, so backticked identifiers stay
  // literal. Bold and italic do recurse, because the documents nest them (`**larger `S`
  // is better**`), and a single non-recursive pass leaves the inner markers on screen.
  // A lone `*` used as a footnote marker is left alone: opening a run requires a
  // non-space after it and a closing `*` on the same line.
  const CODE = /`[^`]+`/;
  const BOLD = /\*\*(?!\s)([\s\S]+?)\*\*/;
  const ITALIC = /(?<![\w*])\*(?!\s)([^*\n]+?)\*(?![\w*])/;
  const LINK = /\[([^\]]+)\]\(([^)]+)\)/;

  const markup = (s: string, keyBase: string, depth = 0): React.ReactNode[] => {
    const parts: React.ReactNode[] = [];
    if (depth > 4) return [s];
    const re = new RegExp(
      `(${CODE.source}|${BOLD.source}|${ITALIC.source}|${LINK.source})`, "g");
    let last = 0, m: RegExpExecArray | null;
    while ((m = re.exec(s))) {
      if (m.index > last) parts.push(s.slice(last, m.index));
      const t = m[0];
      const k = `${keyBase}-${parts.length}`;
      if (t.startsWith("`"))
        parts.push(<code key={k} className="mono rounded px-1 text-[0.95em]" style={{ background: "rgb(0 0 0 / 0.4)", color: "var(--quantum)" }}>{t.slice(1, -1)}</code>);
      else if (t.startsWith("**"))
        parts.push(<strong key={k} className="on-glass">{markup(t.slice(2, -2), k, depth + 1)}</strong>);
      else if (t.startsWith("*"))
        parts.push(<em key={k} className="italic" style={{ color: "var(--text)" }}>{markup(t.slice(1, -1), k, depth + 1)}</em>);
      else {
        const mm = LINK.exec(t)!;
        parts.push(<a key={k} href={mm[2]} target="_blank" rel="noreferrer" className="underline" style={{ color: "var(--quantum)" }}>{markup(mm[1], k, depth + 1)}</a>);
      }
      last = m.index + t.length;
    }
    if (last < s.length) parts.push(s.slice(last));
    return parts;
  };

  const inline = (s: string): React.ReactNode =>
    splitInlineMath(s).map((seg, i) =>
      seg.math
        ? <MathInline key={`m${i}`} tex={seg.value} />
        : <span key={`t${i}`}>{markup(seg.value, `t${i}`)}</span>
    );

  while (i < lines.length) {
    const line = lines[i];

    // Display math. Handled before tables/paragraphs so a formula containing | or #
    // is never reinterpreted as Markdown structure.
    if (line.trim().startsWith("$$")) {
      const buf: string[] = [];
      const firstLine = line.trim();
      if (firstLine.length > 4 && firstLine.endsWith("$$")) {
        buf.push(firstLine.slice(2, -2));
        i++;
      } else {
        buf.push(firstLine.slice(2));
        i++;
        while (i < lines.length && !lines[i].trim().endsWith("$$")) { buf.push(lines[i]); i++; }
        if (i < lines.length) { buf.push(lines[i].trim().replace(/\$\$$/, "")); i++; }
      }
      out.push(<MathBlock key={key++} tex={buf.join("\n").trim()} />);
      continue;
    }

    if (/^\|/.test(line) && i + 1 < lines.length && /^\|[\s:|-]+\|/.test(lines[i + 1])) {
      const header = line.split("|").slice(1, -1).map((s) => s.trim());
      i += 2;
      const rows: string[][] = [];
      while (i < lines.length && /^\|/.test(lines[i])) {
        rows.push(lines[i].split("|").slice(1, -1).map((s) => s.trim()));
        i++;
      }
      out.push(
        <div key={key++} className="overflow-auto my-3">
          <table className="w-full text-sm">
            <thead><tr>{header.map((h, j) => <th key={j} className="th">{inline(h)}</th>)}</tr></thead>
            <tbody>
              {rows.map((r, j) => (
                <tr key={j}>{r.map((c, k) => <td key={k} className="td align-top">{inline(c)}</td>)}</tr>
              ))}
            </tbody>
          </table>
        </div>
      );
      continue;
    }

    if (line.startsWith("```")) {
      i++;
      const buf: string[] = [];
      while (i < lines.length && !lines[i].startsWith("```")) { buf.push(lines[i]); i++; }
      i++;
      out.push(<pre key={key++} className="bg-ink rounded-lg p-3 my-3 overflow-auto text-xs mono">{buf.join("\n")}</pre>);
      continue;
    }

    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const lvl = h[1].length;
      const cls = lvl === 1 ? "text-2xl font-bold mt-6 mb-3"
        : lvl === 2 ? "text-xl font-semibold mt-6 mb-2 text-accent"
        : lvl === 3 ? "text-base font-semibold mt-4 mb-2" : "text-sm font-semibold mt-3 mb-1";
      out.push(<div key={key++} className={cls}>{inline(h[2])}</div>);
      i++; continue;
    }

    if (/^[-*]\s+/.test(line) || /^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      const ordered = /^\d+\.\s+/.test(line);
      // A wrapped item continues on an unmarked line. Without absorbing those, the list
      // ends at the first wrap and the next marker opens a fresh <ol> — which is why
      // every entry would otherwise be numbered "1.".
      const isMarker = (l: string) => /^\s*([-*]|\d+\.)\s+/.test(l);
      const endsItem = (l: string) =>
        !l.trim() || isMarker(l) || /^(#{1,6})\s/.test(l) || /^\|/.test(l) ||
        l.startsWith("```") || l.trim().startsWith("$$") || l.startsWith(">") ||
        /^(---+|\*\*\*+)$/.test(l.trim());
      while (i < lines.length && isMarker(lines[i])) {
        let item = lines[i].replace(/^\s*([-*]|\d+\.)\s+/, "");
        i++;
        while (i < lines.length && !endsItem(lines[i])) {
          item += " " + lines[i].trim(); i++;
        }
        items.push(item);
      }
      const L = ordered ? "ol" : "ul";
      out.push(
        <L key={key++} className={`${ordered ? "list-decimal" : "list-disc"} pl-5 my-2 space-y-1 text-sm`}>
          {items.map((t, j) => <li key={j}>{inline(t)}</li>)}
        </L>
      );
      continue;
    }

    if (line.startsWith(">")) {
      out.push(
        <blockquote key={key++} className="border-l-2 border-warn pl-3 my-2 text-sm text-slate-300">
          {inline(line.replace(/^>\s?/, ""))}
        </blockquote>
      );
      i++; continue;
    }

    if (/^(---+|\*\*\*+)$/.test(line.trim())) {
      out.push(<hr key={key++} className="border-line my-4" />); i++; continue;
    }

    if (!line.trim()) { i++; continue; }

    // Accumulate the whole paragraph before inline parsing: emphasis and links in the
    // source documents wrap across lines, and a line-at-a-time parser would leave the
    // markers visible.
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{1,6})\s/.test(lines[i]) &&
      !/^\|/.test(lines[i]) &&
      !lines[i].startsWith("```") &&
      !lines[i].trim().startsWith("$$") &&
      !lines[i].startsWith(">") &&
      !/^[-*]\s+/.test(lines[i]) &&
      !/^\d+\.\s+/.test(lines[i]) &&
      !/^(---+|\*\*\*+)$/.test(lines[i].trim())
    ) {
      para.push(lines[i].trim());
      i++;
    }
    out.push(
      <p key={key++} className="text-sm leading-relaxed my-2 text-slate-300">
        {inline(para.join(" "))}
      </p>
    );
  }
  return <div>{out}</div>;
}

export default function Research() {
  const [data, setData] = useState<Dict | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [tab, setTab] = useState<
    "review" | "papers" | "audit" | "data" | "ml" | "extrapolation"
  >("review");

  useEffect(() => { getResearch().then(setData).catch((e) => setErr(errorMessage(e))); }, []);

  if (err) return <NotEvaluated what="Research documents" reason={err} />;
  if (!data) return <div className="card text-sm text-muted">Loading research documents…</div>;

  const cur = data.curation_report;
  const ml = data.ml_training_report;
  const extrap = data.mutation_extrapolation_report;

  const TABS = [
    { id: "review", label: "Literature review" },
    { id: "papers", label: "Bibliography" },
    { id: "audit", label: "Scientific audit" },
    { id: "data", label: "Dataset curation" },
    { id: "ml", label: "Model validation" },
    { id: "extrapolation", label: "Mutation extrapolation" },
  ] as const;

  return (
    <div>
      <div className="flex gap-2 mb-6 flex-wrap">
        {TABS.map((t) => (
          <button key={t.id} onClick={() => setTab(t.id)}
            className={`btn text-sm ${tab === t.id ? "bg-accent/20 text-accent border border-accent" : "btn-ghost"}`}>
            {t.label}
          </button>
        ))}
      </div>

      {tab === "review" && (
        data.literature_review
          ? <div className="card"><Markdown text={data.literature_review} /></div>
          : <NotEvaluated what="literature_review.md" reason="file not found on the server" />
      )}
      {tab === "papers" && (
        data.papers
          ? <div className="card"><Markdown text={data.papers} /></div>
          : <NotEvaluated what="papers.md" reason="file not found on the server" />
      )}

      {tab === "audit" && (
        data.scientific_audit
          ? <div className="card"><Markdown text={data.scientific_audit} /></div>
          : <NotEvaluated what="scientific_audit.md" reason="file not found on the server" />
      )}

      {tab === "data" && (cur ? (
        <>
          <Section title="Dataset provenance">
            <div className="card text-sm space-y-2">
              <div><span className="text-muted">source: </span>{cur.source}</div>
              <div><span className="text-muted">file: </span><span className="mono text-xs">{cur.source_file}</span></div>
              <div><span className="text-muted">length window: </span>
                <span className="mono">{JSON.stringify(cur.length_window)}</span></div>
            </div>
          </Section>
          <Section title="Label extraction"
                   subtitle="DRAMP stores concentrations inside prose with heterogeneous units. Rows that could not be parsed are dropped and counted here rather than guessed at.">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Rows in release" value={fmtInt(cur.parse_stats?.rows_total)} />
              <Stat label="Invalid sequence" value={fmtInt(cur.parse_stats?.invalid_sequence)} />
              <Stat label="Outside length window" value={fmtInt(cur.parse_stats?.length_filtered)} />
              <Stat label="MIC parsed" value={fmtInt(cur.parse_stats?.mic_parsed)} />
              <Stat label="Hemolysis: no data" value={fmtInt(cur.parse_stats?.hemo_nodata)} />
              <Stat label="Hemolysis: dose parsed" value={fmtInt(cur.parse_stats?.hemo_dose_parsed)} />
              <Stat label="Hemolysis: qualitative" value={fmtInt(cur.parse_stats?.hemo_qualitative)} />
              <Stat label="Hemolysis: unparseable" value={fmtInt(cur.parse_stats?.hemo_unparseable)} tone="warn" />
            </div>
          </Section>
          <Section title="Targets">
            <div className="grid md:grid-cols-2 gap-4">
              {[["activity", cur.activity], ["hemolysis (regression)", cur.hemolysis_regression]].map(([name, t]: any) => (
                <div key={name} className="card">
                  <div className="label mb-2">{name}</div>
                  <div className="mono text-xs text-accent2 mb-3">{t?.target}</div>
                  <div className="grid grid-cols-2 gap-2">
                    <Stat label="Sequences" value={fmtInt(t?.n_unique_sequences)} />
                    <Stat label="mean ± sd" value={`${fmt(t?.y_mean, 2)} ± ${fmt(t?.y_std, 2)}`} />
                    <Stat label="min" value={fmt(t?.y_min, 2)} />
                    <Stat label="max" value={fmt(t?.y_max, 2)} />
                  </div>
                  {t?.caveat && <p className="text-xs text-warn mt-3">{t.caveat}</p>}
                  {t?.aggregation_across_organisms && (
                    <p className="text-xs text-muted mt-2">
                      aggregation: {t.aggregation_across_organisms}
                    </p>
                  )}
                </div>
              ))}
            </div>
            <div className="card mt-4">
              <div className="label mb-2">Hemolysis classification (cross-check only)</div>
              <div className="text-xs text-muted mb-2">{cur.hemolysis_classification?.labels_from}</div>
              <div className="grid grid-cols-3 gap-2">
                <Stat label="Sequences" value={fmtInt(cur.hemolysis_classification?.n_unique_sequences)} />
                <Stat label="Positive" value={fmtInt(cur.hemolysis_classification?.n_positive)} />
                <Stat label="Negative" value={fmtInt(cur.hemolysis_classification?.n_negative)} />
              </div>
            </div>
          </Section>
          <Section title="Known limitations">
            <div className="card">
              <ul className="text-sm text-slate-300 space-y-2 list-disc pl-5">
                {(cur.known_limitations ?? []).map((l: string, i: number) => <li key={i}>{l}</li>)}
              </ul>
            </div>
          </Section>
        </>
      ) : <NotEvaluated what="Dataset curation report"
                        reason="run: python -m backend.data.curate" />)}

      {tab === "extrapolation" && (extrap ? (
        <>
          <Section
            title="Can the models predict the effect of a mutation?"
            subtitle="The headline metrics elsewhere measure ABSOLUTE prediction accuracy. This measures something different and more directly relevant: given two real, independently assayed peptides differing by 1-3 residues and both held out from training, does the model predict the direction and size of the change? No synthetic mutants and no retraining are involved."
          >
            {Object.entries<any>(extrap).map(([target, r]) => {
              const h = r.headline ?? {};
              const same = r.by_study_provenance?.same_study ?? {};
              const cross = r.by_study_provenance?.cross_study ?? {};
              const tone =
                h.verdict === "USABLE" ? "good"
                : h.verdict === "WEAK BUT REAL" ? "warn"
                : "bad";
              return (
                <div key={target} className="mb-6">
                  <div className="card mb-3">
                    <div className="flex items-center gap-3 mb-2">
                      <span className="label">{target}</span>
                      <span className={`pill ${
                        tone === "good" ? "bg-accent/15 text-accent"
                        : tone === "warn" ? "bg-warn/15 text-warn"
                        : "bg-bad/15 text-bad"}`}>
                        {h.verdict ?? "not evaluated"}
                      </span>
                    </div>
                    <p className="text-sm text-slate-200">{h.statement}</p>
                  </div>

                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                    <Stat label="Held-out pairs" value={fmtInt(r.n_pairs)}
                          sub={`from ${fmtInt(r.n_test_sequences)} test sequences`} />
                    <Stat label="Same-study pairs" value={fmtInt(same.n_pairs)}
                          sub="the fair comparison" />
                    <Stat label="Directional accuracy" value={
                            same.directional_accuracy != null
                              ? `${(100 * same.directional_accuracy).toFixed(1)}%` : "—"}
                          tone={tone === "good" ? "good" : tone === "warn" ? "warn" : "bad"}
                          sub="50% = chance" />
                    <Stat label="Spearman ρ" value={fmt(same.spearman_rho, 3)}
                          sub="on signed change" />
                  </div>

                  <div className="card overflow-auto mb-3">
                    <div className="label mb-2">By measurement provenance</div>
                    <table className="w-full">
                      <thead><tr>
                        <th className="th">stratum</th><th className="th">pairs</th>
                        <th className="th">delta MAE</th><th className="th">"no change" baseline</th>
                        <th className="th">Pearson</th><th className="th">Spearman</th>
                        <th className="th">directional</th>
                      </tr></thead>
                      <tbody>
                        {[["same study", same], ["cross study", cross]].map(([name, v]: any) => (
                          <tr key={name} className={name === "same study" ? "bg-accent/5" : ""}>
                            <td className="td font-medium">{name}</td>
                            <td className="td mono">{fmtInt(v.n_pairs)}</td>
                            <td className="td mono">{fmt(v.mae, 4)}</td>
                            <td className="td mono text-muted">{fmt(v.zero_baseline_mae, 4)}</td>
                            <td className="td mono">{fmt(v.pearson_r, 3)}</td>
                            <td className="td mono">{fmt(v.spearman_rho, 3)}</td>
                            <td className="td mono">
                              {v.directional_accuracy != null
                                ? `${(100 * v.directional_accuracy).toFixed(1)}%` : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="text-xs text-muted mt-3">{r.study_provenance_meaning}</p>
                  </div>

                  <div className="card mb-3">
                    <div className="label mb-2">Pooled figure, and why it misleads</div>
                    <div className="text-sm">
                      Pooling both strata would report directional accuracy{" "}
                      <span className="mono">
                        {h.pooled_would_have_said?.directional_accuracy != null
                          ? `${(100 * h.pooled_would_have_said.directional_accuracy).toFixed(1)}%`
                          : "—"}
                      </span>{" "}
                      and Spearman{" "}
                      <span className="mono">{fmt(h.pooled_would_have_said?.spearman_rho, 3)}</span>.
                    </div>
                    <p className="text-xs text-muted mt-2">
                      {h.pooled_would_have_said?.why_it_differs}
                    </p>
                  </div>

                  <div className="card">
                    <div className="label mb-2">Error by number of mutations</div>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries<any>(r.by_diff_count ?? {}).map(([d, rec]) => (
                        <span key={d} className="pill bg-panel2 mono text-xs">
                          {d} mut · n={rec.n_pairs} · MAE {fmt(rec.mae, 3)}
                        </span>
                      ))}
                    </div>
                    <p className="text-xs text-warn mt-3">{h.caveat_on_mae}</p>
                  </div>
                </div>
              );
            })}
          </Section>
        </>
      ) : (
        <NotEvaluated
          what="Mutation extrapolation report"
          reason="run: python -m backend.models.mutation_extrapolation"
        />
      ))}

      {tab === "ml" && (ml ? (
        <>
          {Object.entries<any>(ml).map(([name, r]) => (
            <Section key={name} title={`${name} model`} subtitle={r.target_definition}>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                <Stat label="Sequences" value={fmtInt(r.n_sequences)} />
                <Stat label="Features" value={fmtInt(r.n_features)} />
                <Stat label="Identity clusters" value={fmtInt(r.redundancy?.n_clusters)}
                      sub={`at ${r.redundancy?.identity_threshold} identity`} />
                <Stat label="Exact duplicates removed" value={fmtInt(r.redundancy?.n_exact_duplicates)} />
              </div>
              <div className="card mb-3">
                <div className="label mb-2">Held-out test (cluster-level split)</div>
                <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                  <Stat label="n" value={fmtInt(r.final_model?.metrics_heldout_test?.n)} />
                  <Stat label="MAE" value={fmt(r.final_model?.metrics_heldout_test?.mae, 3)} />
                  <Stat label="RMSE" value={fmt(r.final_model?.metrics_heldout_test?.rmse, 3)} />
                  <Stat label="RMSE, mean baseline"
                        value={fmt(r.final_model?.metrics_heldout_test?.rmse_mean_baseline, 3)}
                        sub="the bar a model must clear" />
                  <Stat label="Spearman ρ" value={fmt(r.final_model?.metrics_heldout_test?.spearman_rho, 3)}
                        tone={(r.final_model?.metrics_heldout_test?.spearman_rho ?? 0) > 0.4 ? "good" : "warn"} />
                </div>
              </div>
              <div className="card mb-3">
                <div className="label mb-2">Leakage control</div>
                <div className="text-sm space-y-1">
                  <div>{r.split_cluster?.strategy} · train {fmtInt(r.split_cluster?.n_train)} /
                    val {fmtInt(r.split_cluster?.n_val)} / test {fmtInt(r.split_cluster?.n_test)}</div>
                  <div className="text-xs text-muted">
                    optimism gap (RMSE): cluster split {fmt(r.optimism_gap_rmse?.cluster_split_test_rmse, 3)} vs
                    random split {fmt(r.optimism_gap_rmse?.random_split_test_rmse, 3)}.
                    {" "}{r.optimism_gap_rmse?.interpretation}
                  </div>
                  <div className="text-xs text-warn">{r.metrics_random_split?.WARNING}</div>
                </div>
              </div>
              {r.cross_validation?.rmse_mean !== undefined && (
                <div className="card mb-3">
                  <div className="label mb-2">{r.cross_validation.scheme}</div>
                  <div className="grid grid-cols-3 gap-3">
                    <Stat label="MAE" value={`${fmt(r.cross_validation.mae_mean, 3)} ± ${fmt(r.cross_validation.mae_std, 3)}`} />
                    <Stat label="RMSE" value={`${fmt(r.cross_validation.rmse_mean, 3)} ± ${fmt(r.cross_validation.rmse_std, 3)}`} />
                    <Stat label="Spearman" value={fmt(r.cross_validation.spearman_mean, 3)} />
                  </div>
                </div>
              )}
              <div className="card mb-3">
                <div className="label mb-2">Model selection</div>
                <div className="text-xs text-muted">{r.model_selection?.selected_on}</div>
                <div className="text-sm mt-1">{r.model_selection?.note}</div>
              </div>
              {r.bootstrap_ensemble && (
                <div className="card mb-3">
                  <div className="label mb-2">Prediction uncertainty</div>
                  <div className="grid grid-cols-2 gap-3">
                    <Stat label="Ensemble models" value={fmtInt(r.bootstrap_ensemble.n_models)} />
                    <Stat label="Mean predictive sd on test"
                          value={fmt(r.bootstrap_ensemble.mean_predictive_std_on_test, 4)} />
                  </div>
                  <p className="text-xs text-warn mt-2">{r.bootstrap_ensemble.purpose}</p>
                </div>
              )}
              {r.top_features && (
                <div className="card">
                  <div className="label mb-2">Top features by gain</div>
                  <div className="flex flex-wrap gap-2">
                    {r.top_features.slice(0, 12).map((f: any) => (
                      <span key={f.feature} className="pill bg-panel2 mono text-xs">
                        {f.feature} · {fmt(f.gain_importance, 3)}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </Section>
          ))}
        </>
      ) : <NotEvaluated what="ML training report"
                        reason="run: python -m backend.models.train" />)}
    </div>
  );
}
