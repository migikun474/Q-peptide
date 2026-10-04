import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { NavLink, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { animate } from "motion";
import { RunProvider, useRun } from "./lib/store";
import { SPRING, prefersReducedMotion } from "./lib/motion";
import Dashboard from "./pages/Dashboard";
import Optimize from "./pages/Optimize";
import Landscape from "./pages/Landscape";
import Qubo from "./pages/Qubo";
import Quantum from "./pages/Quantum";
import Benchmark from "./pages/Benchmark";
import Results from "./pages/Results";
import Research from "./pages/Research";

const NAV = [
  { to: "/", label: "Overview" },
  { to: "/optimize", label: "Optimize" },
  { to: "/landscape", label: "Landscape" },
  { to: "/qubo", label: "QUBO" },
  { to: "/quantum", label: "Quantum" },
  { to: "/benchmark", label: "Benchmark" },
  { to: "/results", label: "Results" },
  { to: "/research", label: "Research" },
];

function Dot({ ok }: { ok: boolean }) {
  return (
    <span className="relative inline-flex h-1.5 w-1.5">
      <span
        className="inline-flex h-1.5 w-1.5 rounded-full"
        style={{ background: ok ? "var(--bio)" : "var(--bad)" }}
      />
    </span>
  );
}

function StatusChips({ compact = false }: { compact?: boolean }) {
  const { health, runId } = useRun();
  if (!health) {
    return (
      <span className="pill" style={{ background: "rgb(251 113 133 / 0.12)", color: "var(--bad)" }}>
        <Dot ok={false} /> offline
      </span>
    );
  }
  return (
    <div className="flex items-center gap-1.5">
      <span className="pill" style={{
        background: health.models_loaded ? "rgb(94 234 212 / 0.1)" : "rgb(251 113 133 / 0.12)",
        color: health.models_loaded ? "var(--bio)" : "var(--bad)",
      }}>
        <Dot ok={health.models_loaded} /> models
      </span>
      <span className="pill" style={{
        background: health.quantum_backend_available ? "rgb(125 211 252 / 0.1)" : "rgb(251 113 133 / 0.12)",
        color: health.quantum_backend_available ? "var(--quantum)" : "var(--bad)",
      }}>
        <Dot ok={health.quantum_backend_available} /> aer
      </span>
      <span className="pill" style={{
        background: health.ibm_credentials_present ? "rgb(167 139 250 / 0.12)" : "rgb(255 255 255 / 0.04)",
        color: health.ibm_credentials_present ? "var(--violet)" : "var(--text-faint)",
      }}>
        qpu {health.ibm_credentials_present ? "live" : "—"}
      </span>
      {!compact && runId && (
        <span className="pill mono hidden xl:inline-flex"
              style={{ background: "rgb(255 255 255 / 0.04)", color: "var(--text-faint)" }}>
          {runId.length > 14 ? runId.slice(0, 12) + "…" : runId}
        </span>
      )}
    </div>
  );
}

/** A single highlight that springs between tabs, rather than a border that blinks. */
function DesktopNav() {
  const location = useLocation();
  const listRef = useRef<HTMLDivElement | null>(null);
  const pillRef = useRef<HTMLDivElement | null>(null);
  const settled = useRef(false);

  useLayoutEffect(() => {
    const list = listRef.current;
    const pill = pillRef.current;
    if (!list || !pill) return;
    const active = list.querySelector<HTMLElement>('a[aria-current="page"]');
    if (!active) { pill.style.opacity = "0"; return; }

    const target = { x: active.offsetLeft, width: active.offsetWidth };
    pill.style.opacity = "1";
    if (!settled.current || prefersReducedMotion()) {
      pill.style.transform = `translateX(${target.x}px)`;
      pill.style.width = `${target.width}px`;
      settled.current = true;
      return;
    }
    // Springs start from the live on-screen value, so an interrupted switch
    // continues from where it actually is instead of jumping.
    animate(pill, { transform: `translateX(${target.x}px)`, width: `${target.width}px` },
            SPRING.ui);
  }, [location.pathname]);

  return (
    <div ref={listRef} className="relative hidden items-center gap-0.5 lg:flex">
      <div
        ref={pillRef}
        aria-hidden
        className="pointer-events-none absolute inset-y-1 left-0 rounded-full"
        style={{
          background: "rgb(255 255 255 / 0.07)",
          boxShadow: "inset 0 0 0 1px rgb(255 255 255 / 0.09)",
          opacity: 0,
        }}
      />
      {NAV.map((n) => (
        <NavLink
          key={n.to}
          to={n.to}
          end={n.to === "/"}
          className="relative z-10 whitespace-nowrap rounded-full px-3.5 py-1.5 text-[13px] font-semibold tracking-[-0.01em]"
          style={({ isActive }) => ({
            color: isActive ? "var(--text)" : "var(--text-faint)",
            transition: "color 420ms var(--ease)",
          })}
        >
          {n.label}
        </NavLink>
      ))}
    </div>
  );
}

/** Hamburger whose bars rotate into an X, plus a full-bleed staggered overlay. */
function MobileNav() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  useEffect(() => { setOpen(false); }, [location.pathname]);
  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [open]);

  const bar =
    "absolute left-0 h-[1.5px] w-5 rounded-full bg-current transition-transform duration-[560ms] ease-fluid";

  return (
    <>
      <button
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="relative grid h-9 w-9 place-items-center rounded-full lg:hidden"
        style={{ background: "rgb(255 255 255 / 0.05)", boxShadow: "inset 0 0 0 1px rgb(255 255 255 / 0.1)" }}
      >
        <span className="relative block h-3 w-5">
          <span className={bar} style={{ top: open ? "5.5px" : "1px", transform: open ? "rotate(45deg)" : "none" }} />
          <span className={bar} style={{ top: open ? "5.5px" : "10px", transform: open ? "rotate(-45deg)" : "none" }} />
        </span>
      </button>

      <div
        className="fixed inset-0 z-40 lg:hidden"
        style={{
          background: "rgb(5 5 5 / 0.86)",
          backdropFilter: "blur(32px)",
          WebkitBackdropFilter: "blur(32px)",
          opacity: open ? 1 : 0,
          pointerEvents: open ? "auto" : "none",
          transition: "opacity 560ms var(--ease)",
        }}
      >
        <nav className="flex h-full flex-col justify-center gap-1 px-8">
          {NAV.map((n, i) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className="py-2.5 text-[2rem] font-bold tracking-[-0.035em]"
              style={({ isActive }) => ({
                color: isActive ? "var(--bio)" : "var(--text)",
                opacity: open ? 1 : 0,
                transform: open ? "translateY(0)" : "translateY(28px)",
                transition: `opacity 620ms var(--ease) ${60 + i * 45}ms, transform 620ms var(--ease) ${60 + i * 45}ms`,
              })}
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </>
  );
}

function Shell() {
  const mainRef = useRef<HTMLElement | null>(null);
  const location = useLocation();

  useEffect(() => {
    const el = mainRef.current;
    if (!el || prefersReducedMotion()) return;
    animate(el, { opacity: [0, 1], transform: ["translateY(10px)", "translateY(0px)"] },
            SPRING.snap);
    window.scrollTo({ top: 0, behavior: "instant" as ScrollBehavior });
  }, [location.pathname]);

  return (
    <div className="min-h-[100dvh]">
      {/* Floating island: detached from the top edge, width-to-content. */}
      <header className="sticky top-0 z-30 px-4 pt-5">
        <div className="glass-chrome mx-auto flex w-full max-w-[1340px] items-center justify-between gap-5 rounded-full py-2 pl-5 pr-2.5">
          <NavLink to="/" className="flex shrink-0 items-baseline gap-2">
            <span className="text-[17px] font-extrabold tracking-[-0.045em]">
              Q<span className="text-bio">·</span>Peptide
            </span>
          </NavLink>

          <DesktopNav />

          <div className="flex items-center gap-2.5">
            <div className="hidden sm:block"><StatusChips /></div>
            <MobileNav />
          </div>
        </div>
      </header>

      <main ref={mainRef} className="mx-auto max-w-[1340px] px-4 pb-24 pt-10 md:px-6 md:pt-14">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/optimize" element={<Optimize />} />
          <Route path="/landscape" element={<Landscape />} />
          <Route path="/qubo" element={<Qubo />} />
          <Route path="/quantum" element={<Quantum />} />
          <Route path="/benchmark" element={<Benchmark />} />
          <Route path="/results" element={<Results />} />
          <Route path="/research" element={<Research />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>

      <footer className="mx-auto max-w-[1340px] px-6 pb-16 text-center text-[11px] leading-relaxed text-faint">
        Computational design tool · every activity and hemolysis value shown is a model
        prediction, not an experimental result
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <RunProvider>
      <Shell />
    </RunProvider>
  );
}
