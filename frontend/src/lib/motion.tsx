import { animate, inView } from "motion";
import React, { useEffect, useRef } from "react";

/**
 * Motion primitives.
 *
 * Two rules from Apple's fluid-interface work drive everything here:
 *
 *  - Default springs are critically damped (bounce 0). Overshoot is reserved
 *    for motion the user physically initiated -- a flick or a drag release.
 *    Bounce on something that merely appeared reads as a toy.
 *  - Glass surfaces MATERIALISE: blur and scale animate together so the
 *    surface arrives as a real material, rather than a flat opacity fade.
 *
 * Reduced motion is handled once, here, rather than at each call site.
 */

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Apple's documented spring pairs, named by intent rather than by numbers. */
export const SPRING = {
  /** Default UI motion. Critically damped: arrives and settles, no bounce. */
  ui: { type: "spring", bounce: 0, duration: 0.4 },
  /** Snappier variant for small elements and hover affordances. */
  snap: { type: "spring", bounce: 0, duration: 0.28 },
  /** Momentum interactions only -- a release, a flick, a throw. */
  momentum: { type: "spring", bounce: 0.22, duration: 0.4 },
  /** Sheets and drawers. */
  sheet: { type: "spring", bounce: 0.18, duration: 0.3 },
} as const;

interface RevealProps {
  children: React.ReactNode;
  /** Stagger index; converted to a small delay so groups cascade. */
  index?: number;
  className?: string;
  /** Vertical travel in px. Kept small -- large travel is what makes motion sickening. */
  y?: number;
  as?: keyof React.JSX.IntrinsicElements;
}

/**
 * Reveals content when it scrolls into view, materialising rather than fading.
 *
 * Under reduced motion this degrades to a short opacity cross-fade with no
 * transform and no blur -- the comprehension cue survives, the vestibular
 * trigger does not.
 */
export function Reveal({
  children,
  index = 0,
  className,
  y = 14,
  as = "div",
}: RevealProps) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    // Once an element has revealed, it stays revealed. Async data landing (a fetch
    // resolving, a parent re-render) must not re-hide content the user is already
    // reading. The flag lives on the DOM node, so it survives React re-renders that
    // keep the same element.
    if (el.dataset.revealed === "true") return;

    const reduced = prefersReducedMotion();
    const delay = Math.min(index * 0.04, 0.18);

    if (reduced) {
      el.style.opacity = "0";
      const stop = inView(el, () => {
        el.dataset.revealed = "true";
        animate(el, { opacity: 1 }, { duration: 0.2, delay });
      }, { amount: 0.1 });
      return () => stop();
    }

    el.style.opacity = "0";
    el.style.transform = `translateY(${y}px) scale(0.985)`;
    el.style.filter = "blur(6px)";
    el.style.willChange = "transform, opacity, filter";

    const stop = inView(
      el,
      () => {
        el.dataset.revealed = "true";
        animate(
          el,
          { opacity: 1, transform: "translateY(0px) scale(1)", filter: "blur(0px)" },
          { ...SPRING.ui, delay },
        ).finished.then(() => {
          // Release the compositor hint and any stale filter once settled.
          el.style.willChange = "auto";
          el.style.filter = "";
        });
      },
      { amount: 0.1 },
    );
    return () => stop();
  }, [index, y]);

  return React.createElement(
    as,
    { ref: ref as never, className },
    children,
  );
}

/**
 * Press feedback bound to pointer-DOWN, not click.
 *
 * Waiting for pointer-up to acknowledge a press is the single most common way
 * a web UI feels dead. The scale is applied immediately and released on up or
 * cancel, including when the pointer leaves the element mid-press.
 */
export function usePress(scale = 0.97) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || prefersReducedMotion()) return;

    const down = () => animate(el, { scale }, SPRING.snap);
    const up = () => animate(el, { scale: 1 }, SPRING.snap);

    el.addEventListener("pointerdown", down);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("pointerleave", up);
    return () => {
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("pointerup", up);
      el.removeEventListener("pointercancel", up);
      el.removeEventListener("pointerleave", up);
    };
  }, [scale]);

  return ref;
}

/**
 * Counts a number up when it first appears.
 *
 * Deliberately short and non-bouncy: this is a comprehension aid that draws the
 * eye to a figure, not a celebration. Skipped entirely under reduced motion,
 * where the final value is simply rendered.
 */
export function useCountUp(value: number, digits = 0, duration = 0.7) {
  const ref = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || !Number.isFinite(value)) return;

    const format = (v: number) =>
      digits > 0
        ? v.toFixed(digits)
        : Math.round(v).toLocaleString();

    if (prefersReducedMotion()) {
      el.textContent = format(value);
      return;
    }

    const stop = inView(el, () => {
      animate(0, value, {
        duration,
        ease: [0.16, 1, 0.3, 1],
        onUpdate: (v: number) => { el.textContent = format(v); },
      });
    }, { amount: 0.4 });
    return () => stop();
  }, [value, digits, duration]);

  return ref;
}
