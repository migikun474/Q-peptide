import { Link } from "react-router-dom";
import { fmt } from "../lib/api";
import { Reveal, useCountUp, usePress } from "../lib/motion";

export function NotEvaluated({ what, reason }: { what: string; reason?: string }) {
  return (
    <div className="card" style={{ borderStyle: "dashed" }}>
      <div className="pill mb-2" style={{ background: "rgb(251 191 36 / 0.12)", color: "var(--warn)" }}>
        Not evaluated
      </div>
      <div className="text-sm on-glass">{what}</div>
      {reason && <div className="mt-1 text-xs text-faint">{reason}</div>}
    </div>
  );
}

export function NoRun() {
  const press = usePress();
  return (
    <div className="card card-raised py-14 text-center">
      <div className="title mb-2">No optimization run loaded</div>
      <p className="mx-auto mb-6 max-w-lg text-sm text-dim">
        Every page here renders results from a real executed run. Nothing is simulated or
        filled in with placeholder values, so there is nothing to show until one completes.
      </p>
      <Link ref={press as never} to="/optimize" className="btn-primary">
        Run an optimization
        <span className="btn-dot">↗</span>
      </Link>
    </div>
  );
}

const TONE: Record<string, string> = {
  good: "var(--bio)",
  bad: "var(--bad)",
  warn: "var(--warn)",
  quantum: "var(--quantum)",
  default: "var(--text)",
};

export function Stat({
  label, value, sub, tone = "default", countTo, digits = 0,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  tone?: "default" | "good" | "bad" | "warn" | "quantum";
  /** When given, the figure counts up on first appearance instead of popping in. */
  countTo?: number;
  digits?: number;
}) {
  const countRef = useCountUp(countTo ?? NaN, digits);
  return (
    <div className="card-tight">
      <div className="label">{label}</div>
      <div className="numeral mt-2.5 text-[26px] font-bold leading-none"
           style={{ color: TONE[tone] }}>
        {countTo !== undefined ? <span ref={countRef}>0</span> : value}
      </div>
      {sub && <div className="mt-2 text-[11px] leading-snug text-faint">{sub}</div>}
    </div>
  );
}

export function Section({
  title, subtitle, children, right, index = 0, eyebrow,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  right?: React.ReactNode;
  index?: number;
  eyebrow?: string;
}) {
  return (
    <Reveal as="section" index={index} className="mb-20 block md:mb-28">
      <div className="mb-6 flex items-start justify-between gap-6">
        <div>
          {eyebrow && <div className="eyebrow mb-3.5">{eyebrow}</div>}
          <h2 className="title">{title}</h2>
          {subtitle && (
            <p className="mt-2 max-w-3xl text-[13.5px] leading-relaxed text-dim">{subtitle}</p>
          )}
        </div>
        {right}
      </div>
      {children}
    </Reveal>
  );
}

export function PassFail({ ok, label }: { ok: boolean | null | undefined; label: string }) {
  if (ok === null || ok === undefined) {
    return (
      <span className="pill" style={{ background: "rgb(251 191 36 / 0.12)", color: "var(--warn)" }}>
        {label}: not evaluated
      </span>
    );
  }
  return (
    <span className="pill" style={{
      background: ok ? "rgb(94 234 212 / 0.12)" : "rgb(251 113 133 / 0.14)",
      color: ok ? "var(--bio)" : "var(--bad)",
    }}>
      {ok ? "✓" : "✗"} {label}
    </span>
  );
}

export function SequenceView({
  sequence, parent, highlight, size = "sm",
}: {
  sequence: string;
  parent?: string;
  highlight?: number[];
  size?: "sm" | "lg";
}) {
  const hi = new Set(highlight ?? []);
  return (
    <div className={`mono break-all ${size === "lg" ? "text-[15px] leading-8" : "leading-7"}`}>
      {sequence.split("").map((ch, i) => {
        const changed = parent ? parent[i] !== ch : false;
        const marked = hi.has(i) || changed;
        return (
          <span
            key={i}
            title={parent && changed
              ? `position ${i + 1}: ${parent[i]} → ${ch}`
              : `position ${i + 1}: ${ch}`}
            className={marked ? "rounded px-0.5 font-semibold" : ""}
            style={marked ? { background: "rgb(94 234 212 / 0.22)", color: "var(--bio)" } : undefined}
          >
            {ch}
          </span>
        );
      })}
    </div>
  );
}

export function Disclaimer() {
  return (
    <div className="card text-xs leading-relaxed text-dim"
         style={{ borderColor: "rgb(251 191 36 / 0.22)", background: "rgb(251 191 36 / 0.04)" }}>
      <span className="font-semibold" style={{ color: "var(--warn)" }}>Computational tool. </span>
      Every activity and hemolysis value shown anywhere in this application is a
      <strong className="on-glass"> model prediction</strong> from machine-learning models
      trained on literature-extracted data. No experimental validation is claimed or
      implied, and experimental validation would be required before drawing any biological
      conclusion.
    </div>
  );
}

export function KeyVal({ data, keys }: { data: Record<string, any>; keys?: string[] }) {
  const entries = (keys ?? Object.keys(data)).filter((k) => data[k] !== undefined);
  return (
    <dl className="grid grid-cols-1 gap-x-6 gap-y-2 text-sm sm:grid-cols-2">
      {entries.map((k) => (
        <div key={k} className="flex justify-between gap-3 border-b pb-1"
             style={{ borderColor: "rgb(255 255 255 / 0.06)" }}>
          <dt className="text-faint">{k.replace(/_/g, " ")}</dt>
          <dd className="mono numeral text-right on-glass">
            {typeof data[k] === "number" ? fmt(data[k]) : String(data[k])}
          </dd>
        </div>
      ))}
    </dl>
  );
}
