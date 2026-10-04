import { useState } from "react";
import { useNavigate } from "react-router-dom";
import type { OptimizeRequest } from "../lib/api";
import { analyzePeptide, errorMessage, fmt, getResults, startOptimize } from "../lib/api";
import { useRun } from "../lib/store";
import { Disclaimer, Section, Stat } from "../components/Common";

const PRESETS: { name: string; seq: string; note: string }[] = [
  { name: "Magainin 2", seq: "GIGKFLHSAKKFGKAFVGEIMNS", note: "classic α-helical AMP; the charge/hydrophobicity trade-off literature is built on its analogues" },
  { name: "Buforin II", seq: "TRSSRAGLQFPVGRVHRLLRK", note: "histone-derived, cationic" },
  { name: "Aurein 1.2", seq: "GLFDIIKKIAESF", note: "short amphipathic helix" },
];

export default function Optimize() {
  const nav = useNavigate();
  const { setRunId, setRun, setStatus, health } = useRun();

  const [seq, setSeq] = useState(PRESETS[0].seq);
  const [k, setK] = useState(3);
  const [mode, setMode] = useState<"at_most_k" | "exactly_k">("at_most_k");
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(1);
  const [targetN, setTargetN] = useState(14);
  const [maxPer, setMaxPer] = useState(2);
  const [depths, setDepths] = useState<number[]>([1, 2, 3]);
  const [optimizers, setOptimizers] = useState<string[]>(["COBYLA", "Powell"]);
  const [shots, setShots] = useState(8192);
  const [noise, setNoise] = useState(true);
  const [hardware, setHardware] = useState(false);

  const [analysis, setAnalysis] = useState<any>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [progress, setProgress] = useState<string | null>(null);

  const doAnalyze = async () => {
    setErr(null); setAnalysis(null);
    try { setAnalysis(await analyzePeptide(seq)); }
    catch (e) { setErr(errorMessage(e)); }
  };

  const toggle = <T,>(arr: T[], v: T, set: (a: T[]) => void) =>
    set(arr.includes(v) ? arr.filter((x) => x !== v) : [...arr, v]);

  const run = async () => {
    setErr(null); setBusy(true); setProgress("submitting…");
    const req: OptimizeRequest = {
      sequence: seq, budget_k: k, budget_mode: mode, alpha, beta,
      target_n: targetN, max_per_position: maxPer,
      qaoa_depths: depths.length ? depths : [1],
      qaoa_optimizers: optimizers.length ? optimizers : ["COBYLA"],
      qaoa_maxiter: 300, qaoa_restarts: 3, shots,
      include_noise: noise, include_hardware: hardware,
      include_surrogate_validation: true,
    };
    try {
      const { optimization_id } = await startOptimize(req);
      setRunId(optimization_id); setStatus("running"); setRun(null);
      setProgress("running the pipeline — exact enumeration, classical baselines, QAOA sweep…");
      // poll
      for (;;) {
        await new Promise((r) => setTimeout(r, 3000));
        const res = await getResults(optimization_id);
        if (res.config) { setRun(res); setStatus("completed"); nav("/results"); return; }
      }
    } catch (e) {
      setErr(errorMessage(e)); setStatus("failed"); setBusy(false); setProgress(null);
    }
  };

  const nVars = targetN + (mode === "at_most_k" ? Math.max(0, Math.ceil(Math.log2(k + 1))) : 0);

  return (
    <div>
      <Section title="Parent peptide">
        <div className="card space-y-4">
          <div>
            <label className="label">Sequence (one-letter codes, length 5–60)</label>
            <input className="input mt-1" value={seq}
                   onChange={(e) => setSeq(e.target.value.toUpperCase())} spellCheck={false} />
            <div className="text-xs text-muted mt-1">{seq.length} residues</div>
          </div>
          <div className="flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button key={p.name} title={p.note} onClick={() => setSeq(p.seq)}
                      className="btn-ghost text-xs py-1">{p.name}</button>
            ))}
            <button onClick={doAnalyze} className="btn-ghost text-xs py-1">Analyze parent</button>
          </div>
          {analysis && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
              <Stat label="Net charge (pH 7.4)" value={fmt(analysis.net_charge_ph74, 2)} />
              <Stat label="Isoelectric point" value={fmt(analysis.isoelectric_point, 2)} />
              <Stat label="Hydrophobic moment" value={fmt(analysis.hydrophobic_moment, 3)} />
              <Stat label="Mean hydrophobicity" value={fmt(analysis.mean_hydrophobicity_kd, 3)} />
              <Stat label="Predicted activity" value={fmt(analysis.model_predicted_activity, 3)}
                    sub="pMIC scale; higher = more active" />
              <Stat label="Predicted hemolysis" value={fmt(analysis.model_predicted_hemolysis, 3)}
                    sub="p-dose scale; higher = MORE hemolytic" />
              <Stat label="Score S(P)" value={fmt(analysis.score, 4)} sub="αÃ − βH̃ at α=β=1" />
              <Stat label="Length" value={analysis.length} />
            </div>
          )}
        </div>
      </Section>

      <Section
        title="Mutation constraint"
        subtitle="These are two mathematically different constraints. The choice is explicit and is never inferred."
      >
        <div className="card space-y-4">
          <div className="grid sm:grid-cols-2 gap-3">
            <button onClick={() => setMode("at_most_k")}
              className={`text-left p-4 rounded-lg border transition-colors ${
                mode === "at_most_k" ? "border-accent bg-accent/10" : "border-line hover:bg-panel2"}`}>
              <div className="font-semibold text-sm">At most K mutations</div>
              <div className="mono text-xs text-accent2 mt-1">Σᵢ xᵢ ≤ K</div>
              <div className="text-xs text-muted mt-2">
                Encoded with integer slack bits s ∈ [0, K] and the penalty P(Σxᵢ + s − K)².
                Fewer than K mutations is allowed, so the optimizer may return a smaller set.
              </div>
            </button>
            <button onClick={() => setMode("exactly_k")}
              className={`text-left p-4 rounded-lg border transition-colors ${
                mode === "exactly_k" ? "border-accent bg-accent/10" : "border-line hover:bg-panel2"}`}>
              <div className="font-semibold text-sm">Exactly K mutations</div>
              <div className="mono text-xs text-accent2 mt-1">Σᵢ xᵢ = K</div>
              <div className="text-xs text-muted mt-2">
                Encoded with the equality penalty P(Σxᵢ − K)² and no slack qubits.
                Returning fewer than K mutations is infeasible.
              </div>
            </button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="label">Mutation budget K</label>
              <input type="number" min={1} max={10} value={k} className="input mt-1"
                     onChange={(e) => setK(Math.max(1, Number(e.target.value)))} />
              {k > 2 && (
                <div className="text-xs text-warn mt-1">
                  K ≥ 3 leaves the regime where the QUBO is an exact model of the ML
                  landscape; the run reports measured fidelity.
                </div>
              )}
            </div>
            <div>
              <label className="label">Candidate mutations N</label>
              <input type="number" min={2} max={22} value={targetN} className="input mt-1"
                     onChange={(e) => setTargetN(Number(e.target.value))} />
              <div className="text-xs text-muted mt-1">≈ {nVars} qubits total</div>
            </div>
            <div>
              <label className="label">Max substitutions per position</label>
              <input type="number" min={1} max={6} value={maxPer} className="input mt-1"
                     onChange={(e) => setMaxPer(Number(e.target.value))} />
              <div className="text-xs text-muted mt-1">&gt;1 creates same-position conflicts</div>
            </div>
            <div>
              <label className="label">Shots</label>
              <input type="number" min={128} step={1024} value={shots} className="input mt-1"
                     onChange={(e) => setShots(Number(e.target.value))} />
            </div>
          </div>
        </div>
      </Section>

      <Section title="Objective weights" subtitle="S(P) = α·Ã(P) − β·H̃(P). Larger S is better. Both terms are standardised with statistics frozen at training time, so α and β are comparable.">
        <div className="card grid grid-cols-1 sm:grid-cols-2 gap-6">
          <div>
            <label className="label">Activity weight α = {alpha.toFixed(2)}</label>
            <input type="range" min={0} max={3} step={0.05} value={alpha} className="w-full mt-2"
                   onChange={(e) => setAlpha(Number(e.target.value))} />
          </div>
          <div>
            <label className="label">Hemolysis penalty β = {beta.toFixed(2)}</label>
            <input type="range" min={0} max={3} step={0.05} value={beta} className="w-full mt-2"
                   onChange={(e) => setBeta(Number(e.target.value))} />
          </div>
        </div>
      </Section>

      <Section title="Quantum settings">
        <div className="card space-y-4">
          <div>
            <div className="label mb-2">QAOA depths p</div>
            <div className="flex gap-2">
              {[1, 2, 3, 4].map((p) => (
                <button key={p} onClick={() => toggle(depths, p, setDepths)}
                  className={`btn text-xs ${depths.includes(p) ? "bg-accent2/20 text-accent2 border border-accent2" : "btn-ghost"}`}>
                  p = {p}
                </button>
              ))}
            </div>
          </div>
          <div>
            <div className="label mb-2">Classical optimizers (both are run rather than assuming COBYLA is best)</div>
            <div className="flex gap-2 flex-wrap">
              {["COBYLA", "Powell", "Nelder-Mead"].map((o) => (
                <button key={o} onClick={() => toggle(optimizers, o, setOptimizers)}
                  className={`btn text-xs ${optimizers.includes(o) ? "bg-accent2/20 text-accent2 border border-accent2" : "btn-ghost"}`}>
                  {o}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap gap-5 pt-1">
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={noise} onChange={(e) => setNoise(e.target.checked)} />
              Run the noise study (Aer depolarizing + thermal)
            </label>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={hardware} disabled={!health?.ibm_credentials_present}
                     onChange={(e) => setHardware(e.target.checked)} />
              Execute on IBM Quantum hardware
              {!health?.ibm_credentials_present && (
                <span className="text-xs text-muted">(no token in environment)</span>
              )}
            </label>
          </div>
        </div>
      </Section>

      {err && <div className="card border-bad/40 bg-bad/5 text-sm text-bad mb-4">{err}</div>}
      {progress && <div className="card border-accent2/30 bg-accent2/5 text-sm mb-4">{progress}</div>}

      <div className="flex items-center gap-3 mb-8">
        <button onClick={run} disabled={busy || !health?.models_loaded} className="btn-primary">
          {busy ? "Running…" : "Run optimization"}
        </button>
        {!health?.models_loaded && (
          <span className="text-xs text-warn">
            Models are not loaded — train them first (see README).
          </span>
        )}
      </div>

      <Disclaimer />
    </div>
  );
}
