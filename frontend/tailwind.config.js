/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Kept as named tokens so existing pages inherit the palette without edits.
        ink: "#070b16",
        panel: "#0e1526",
        panel2: "#161f36",
        line: "#232e4a",
        accent: "#5eead4",   // biology
        accent2: "#7dd3fc",  // quantum
        warn: "#fbbf24",
        bad: "#fb7185",
        muted: "#93a0bd",
      },
      fontFamily: {
        mono: ["ui-monospace", "SF Mono", "SFMono-Regular", "Menlo", "monospace"],
      },
      borderRadius: { xl: "14px", "2xl": "20px" },
    },
  },
  plugins: [],
};
