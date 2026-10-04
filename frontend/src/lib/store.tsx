import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { Dict, HealthInfo } from "./api";
import { getHealth, getResults, listResults } from "./api";

/** One shared run record so every page shows the SAME executed run.
 *  Pages never recompute or fabricate values; if no run exists they say so. */
interface RunState {
  runId: string | null;
  run: Dict | null;
  status: "idle" | "running" | "completed" | "failed";
  error: string | null;
  health: HealthInfo | null;
  setRunId: (id: string | null) => void;
  setRun: (r: Dict | null) => void;
  setStatus: (s: RunState["status"]) => void;
  setError: (e: string | null) => void;
  refreshHealth: () => void;
}

const Ctx = createContext<RunState | null>(null);
const KEY = "qpeptide.runId";

export function RunProvider({ children }: { children: React.ReactNode }) {
  const [runId, setRunId] = useState<string | null>(() => localStorage.getItem(KEY));
  const [run, setRun] = useState<Dict | null>(null);
  const [status, setStatus] = useState<RunState["status"]>("idle");
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);

  const refreshHealth = () => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  };

  useEffect(() => { refreshHealth(); }, []);

  useEffect(() => {
    if (runId) localStorage.setItem(KEY, runId);
    else localStorage.removeItem(KEY);
  }, [runId]);

  // Rehydrate a previously selected run, or fall back to the most recent one on disk.
  useEffect(() => {
    if (run || status === "running") return;
    const load = async (id: string) => {
      try {
        const r = await getResults(id);
        if (r.config) { setRun(r); setStatus("completed"); }
      } catch { /* leave empty; the UI will say no run is loaded */ }
    };
    if (runId) { load(runId); return; }
    listResults()
      .then((d) => {
        const done = d.runs.find((x) => x.status === "completed");
        if (done) { setRunId(done.optimization_id); load(done.optimization_id); }
      })
      .catch(() => {});
  }, [runId]);

  const value = useMemo<RunState>(
    () => ({ runId, run, status, error, health, setRunId, setRun, setStatus, setError, refreshHealth }),
    [runId, run, status, error, health]
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useRun(): RunState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useRun must be used inside RunProvider");
  return v;
}
