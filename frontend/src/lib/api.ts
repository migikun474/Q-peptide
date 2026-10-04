import axios from "axios";

const BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export const api = axios.create({ baseURL: BASE, timeout: 600_000 });

/** Shapes are intentionally loose: the backend record is the source of truth and the
 *  UI renders whatever it actually produced. Anything missing is shown as
 *  "Not evaluated" rather than defaulted to a number. */
export type Dict = Record<string, any>;

export interface HealthInfo {
  status: string;
  models_loaded: boolean;
  activity_model: Dict | null;
  hemolysis_model: Dict | null;
  quantum_backend_available: boolean;
  ibm_credentials_present: boolean;
  versions: Record<string, string>;
  disclaimer: string;
}

export interface OptimizeRequest {
  sequence: string;
  budget_k: number;
  budget_mode: "at_most_k" | "exactly_k";
  alpha: number;
  beta: number;
  target_n: number;
  max_per_position: number;
  qaoa_depths: number[];
  qaoa_optimizers: string[];
  qaoa_maxiter: number;
  qaoa_restarts: number;
  shots: number;
  include_noise: boolean;
  include_hardware: boolean;
  include_surrogate_validation: boolean;
}

export const getHealth = () => api.get<HealthInfo>("/api/health").then((r) => r.data);

export const analyzePeptide = (sequence: string) =>
  api.post<Dict>("/api/peptide/analyze", { sequence }).then((r) => r.data);

export const startOptimize = (req: OptimizeRequest) =>
  api.post<{ optimization_id: string; status: string }>("/api/optimize", req).then((r) => r.data);

export const getResults = (id: string) =>
  api.get<Dict>(`/api/results/${id}`).then((r) => r.data);

export const listResults = () =>
  api.get<{ runs: Dict[]; count: number }>("/api/results").then((r) => r.data);

export const listExperiments = () =>
  api.get<{ experiments: string[] }>("/api/experiments").then((r) => r.data);

export const getExperiment = (name: string) =>
  api.get<Dict>(`/api/experiments/${name}`).then((r) => r.data);

export const getResearch = () => api.get<Dict>("/api/research").then((r) => r.data);

export function errorMessage(e: unknown): string {
  if (axios.isAxiosError(e)) {
    const d = e.response?.data as any;
    if (d?.detail) return typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail);
    if (e.code === "ERR_NETWORK") return "Cannot reach the API. Is the backend running on port 8000?";
    return e.message;
  }
  return String(e);
}

/** Format a number for display, never inventing precision. */
export function fmt(v: unknown, digits = 4): string {
  if (v === null || v === undefined) return "—";
  if (typeof v !== "number" || Number.isNaN(v)) return String(v);
  if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e6)) return v.toExponential(2);
  return v.toFixed(digits);
}

export function fmtInt(v: unknown): string {
  if (typeof v !== "number") return "—";
  return v.toLocaleString();
}
