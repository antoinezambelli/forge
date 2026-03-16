import type { ConfigRow } from "./types";

interface ComparePanelProps {
  a: ConfigRow;
  b: ConfigRow;
  scenarios: string[];
  scenarioAbbrev: Record<string, string>;
  onSwap: () => void;
  onClear: () => void;
}

interface MetricDef {
  key: string;
  label: string;
  fmt: (v: number | null) => string;
  higherBetter: boolean;
}

const METRICS: MetricDef[] = [
  { key: "score", label: "Score", fmt: (v) => v == null ? "\u2014" : `${v.toFixed(1)}%`, higherBetter: true },
  { key: "accuracy", label: "Accuracy", fmt: (v) => v == null ? "\u2014" : `${v.toFixed(1)}%`, higherBetter: true },
  { key: "completeness", label: "Completeness", fmt: (v) => v == null ? "\u2014" : `${v.toFixed(1)}%`, higherBetter: true },
  { key: "efficiency", label: "Efficiency", fmt: (v) => v == null ? "\u2014" : `${v.toFixed(1)}%`, higherBetter: true },
  { key: "wasted", label: "Avg Wasted", fmt: (v) => v == null ? "\u2014" : v.toFixed(1), higherBetter: false },
  { key: "speed", label: "Speed", fmt: (v) => v == null ? "\u2014" : `${v.toFixed(1)}s`, higherBetter: false },
];

function DeltaCell({
  va,
  vb,
  higherBetter,
}: {
  va: number | null;
  vb: number | null;
  higherBetter: boolean;
}) {
  if (va == null || vb == null) {
    return <td className="p-1.5 text-right text-zinc-600">&mdash;</td>;
  }
  const delta = vb - va;
  const sign = delta > 0 ? "+" : "";
  const formatted = sign + (Number.isInteger(delta) ? delta : delta.toFixed(1));

  let cls = "text-zinc-500";
  if (delta !== 0) {
    const isBetter = (delta > 0) === higherBetter;
    cls = isBetter ? "text-emerald-400" : "text-red-400";
  }

  return <td className={`p-1.5 text-right tabular-nums font-medium ${cls}`}>{formatted}</td>;
}

export function ComparePanel({
  a,
  b,
  scenarios,
  scenarioAbbrev,
  onSwap,
  onClear,
}: ComparePanelProps) {
  const getVal = (row: ConfigRow, key: string): number | null => {
    if (key in row.scenarios) return row.scenarios[key];
    return (row as unknown as Record<string, number | null>)[key] ?? null;
  };

  return (
    <div className="mt-6 border border-zinc-800 rounded-lg p-4 max-w-2xl">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold">
          Compare
        </h3>
        <div className="flex gap-2">
          <button
            onClick={onSwap}
            className="text-xs px-2.5 py-1 rounded border border-zinc-700 hover:border-zinc-500 transition-colors"
          >
            Swap A&#8596;B
          </button>
          <button
            onClick={onClear}
            className="text-xs px-2.5 py-1 rounded border border-zinc-700 hover:border-red-500/50 hover:text-red-400 transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      <table className="text-xs w-full border-collapse">
        <thead>
          <tr className="border-b border-zinc-800">
            <th className="p-1.5 text-left text-zinc-500">Metric</th>
            <th className="p-1.5 text-right text-zinc-400 max-w-48 truncate" title={a.label}>
              {a.label}
            </th>
            <th className="p-1.5 text-right text-zinc-500 w-16">Delta</th>
            <th className="p-1.5 text-right text-zinc-400 max-w-48 truncate" title={b.label}>
              {b.label}
            </th>
          </tr>
        </thead>
        <tbody>
          {METRICS.map((m) => {
            const va = getVal(a, m.key);
            const vb = getVal(b, m.key);
            return (
              <tr key={m.key} className="border-b border-zinc-900/50">
                <td className="p-1.5 text-zinc-400">{m.label}</td>
                <td className="p-1.5 text-right tabular-nums">{m.fmt(va)}</td>
                <DeltaCell va={va} vb={vb} higherBetter={m.higherBetter} />
                <td className="p-1.5 text-right tabular-nums">{m.fmt(vb)}</td>
              </tr>
            );
          })}

          {/* Separator */}
          <tr>
            <td colSpan={4} className="py-1">
              <div className="border-t border-zinc-800" />
            </td>
          </tr>

          {/* Per-scenario rows */}
          {scenarios.map((sc) => {
            const va = a.scenarios[sc];
            const vb = b.scenarios[sc];
            const fmtSc = (v: number | null, row: ConfigRow) => {
              if (v != null) return `${v}%`;
              return (row.scenarioRuns?.[sc] ?? 0) === 0 ? "I" : "\u2014";
            };
            return (
              <tr key={sc} className="border-b border-zinc-900/50">
                <td className="p-1.5 text-zinc-500">{scenarioAbbrev[sc] || sc}</td>
                <td className="p-1.5 text-right tabular-nums">{fmtSc(va, a)}</td>
                <DeltaCell va={va} vb={vb} higherBetter={true} />
                <td className="p-1.5 text-right tabular-nums">{fmtSc(vb, b)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
