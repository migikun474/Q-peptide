import { useMemo } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";

/**
 * LaTeX rendering for the research documents.
 *
 * The documents are authored in Markdown with `$$...$$` display math and `$...$` inline
 * math. KaTeX is used rather than MathJax because it renders synchronously and ships a
 * fraction of the weight.
 *
 * `throwOnError: false` makes a malformed expression render in red *in place* rather than
 * taking down the whole page — a broken formula in one document should not blank the
 * Research tab.
 */
function render(tex: string, displayMode: boolean): string {
  try {
    return katex.renderToString(tex, {
      displayMode,
      throwOnError: false,
      strict: false,
      trust: false,
      output: "html",
      macros: {
        // Shorthands the research docs lean on.
        "\\dd": "\\mathrm{d}",
      },
    });
  } catch {
    return "";
  }
}

export function MathBlock({ tex }: { tex: string }) {
  const html = useMemo(() => render(tex, true), [tex]);
  if (!html) {
    return <pre className="mono overflow-auto text-xs text-faint">{tex}</pre>;
  }
  return (
    <div
      className="katex-display-wrap my-5 overflow-x-auto overflow-y-hidden py-1"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

export function MathInline({ tex }: { tex: string }) {
  const html = useMemo(() => render(tex, false), [tex]);
  if (!html) return <code className="mono">{tex}</code>;
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/**
 * Split a run of text on inline `$...$` spans.
 *
 * A `$` only opens math when it is followed by a non-space and the span closes on the
 * same run — that keeps prices and stray dollars from being swallowed as math.
 */
export function splitInlineMath(text: string): Array<{ math: boolean; value: string }> {
  const out: Array<{ math: boolean; value: string }> = [];
  const re = /\$(?!\s)([^$\n]+?)\$/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push({ math: false, value: text.slice(last, m.index) });
    out.push({ math: true, value: m[1] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ math: false, value: text.slice(last) });
  return out;
}
