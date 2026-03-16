import { Fragment } from "react";
import type { ConfigRow, SortState } from "./types";
import type { RowGroup } from "./utils";
import { heatClass, fmtPct } from "./utils";

interface Column {
  key: string;
  label: string;
  sortable?: boolean;
}

interface DataTableProps {
  rows: ConfigRow[];
  scenarios: string[];
  scenarioAbbrev: Record<string, string>;
  sort: SortState;
  onSort: (col: string) => void;
  checked: number[];
  onCompareToggle: (idx: number, on: boolean) => void;
  groups: RowGroup[];
}

const METRIC_COLS: Column[] = [
  { key: "score", label: "Scr%" },
  { key: "accuracy", label: "Acc%" },
  { key: "completeness", label: "Cmp%" },
  { key: "efficiency", label: "Eff%" },
  { key: "wasted", label: "Wst" },
  { key: "speed", label: "Spd" },
  { key: "n", label: "N" },
];

function SortArrow({ col, sort }: { col: string; sort: SortState }) {
  if (sort.col !== col) return null;
  return (
    <span className="ml-0.5 text-emerald-400">
      {sort.asc ? "\u25B2" : "\u25BC"}
    </span>
  );
}

export function DataTable({
  rows,
  scenarios,
  scenarioAbbrev,
  sort,
  onSort,
  checked,
  onCompareToggle,
  groups,
}: DataTableProps) {
  // Pre-compute group separator positions (row index → group label)
  const groupStartAt = new Map<number, string>();
  if (groups.length > 0) {
    let offset = 0;
    for (const g of groups) {
      groupStartAt.set(offset, g.label);
      offset += g.rows.length;
    }
  }

  const totalCols = 2 + METRIC_COLS.length + scenarios.length; // checkbox + label + metrics + scenarios
  return (
    <div className="overflow-x-auto">
      <table className="text-xs whitespace-nowrap border-collapse">
        <thead>
          <tr className="border-b border-zinc-800">
            <th className="p-1.5 w-8" />
            <th
              className="p-1.5 text-left cursor-pointer select-none hover:text-emerald-400 sticky left-0 bg-zinc-950 z-10"
              onClick={() => onSort("label")}
            >
              Model/Backend
              <SortArrow col="label" sort={sort} />
            </th>
            {METRIC_COLS.map((c) => (
              <th
                key={c.key}
                className="p-1.5 text-right cursor-pointer select-none hover:text-emerald-400"
                onClick={() => onSort(c.key)}
              >
                {c.label}
                <SortArrow col={c.key} sort={sort} />
              </th>
            ))}
            {scenarios.map((sc) => (
              <th
                key={sc}
                className="p-1.5 text-right cursor-pointer select-none hover:text-emerald-400"
                onClick={() => onSort(sc)}
                title={sc}
              >
                {scenarioAbbrev[sc] || sc.slice(0, 3)}
                <SortArrow col={sc} sort={sort} />
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const isChecked = checked.includes(i);
            const groupLabel = groupStartAt.get(i);
            return (
              <Fragment key={row.label}>
              {groupLabel != null && (
                <tr className="bg-zinc-900/30">
                  <td colSpan={totalCols} className="px-2 py-1 text-[0.6rem] font-semibold text-zinc-400 uppercase tracking-wider border-t border-zinc-700/50">
                    {groupLabel}
                  </td>
                </tr>
              )}
              <tr
                className={`border-b border-zinc-900 hover:bg-zinc-900/50 transition-colors ${
                  isChecked ? "bg-zinc-800/40" : ""
                }`}
              >
                <td className="p-1.5 text-center">
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={(e) => onCompareToggle(i, e.target.checked)}
                    className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 accent-emerald-500 cursor-pointer"
                  />
                </td>
                <td className="p-1.5 font-mono sticky left-0 bg-zinc-950 z-10">
                  {row.label}
                </td>
                {/* Score */}
                <td className={`p-1.5 text-right tabular-nums ${heatClass(row.score)}`}>
                  {fmtPct(row.score, 1)}
                </td>
                {/* Accuracy */}
                <td className={`p-1.5 text-right tabular-nums ${heatClass(row.accuracy)}`}>
                  {fmtPct(row.accuracy, 1)}
                </td>
                {/* Completeness */}
                <td className={`p-1.5 text-right tabular-nums ${heatClass(row.completeness)}`}>
                  {fmtPct(row.completeness, 1)}
                </td>
                {/* Efficiency */}
                <td className={`p-1.5 text-right tabular-nums ${heatClass(row.efficiency)}`}>
                  {fmtPct(row.efficiency)}
                </td>
                {/* Wasted */}
                <td className="p-1.5 text-right tabular-nums text-zinc-400">
                  {row.wasted.toFixed(1)}
                </td>
                {/* Speed */}
                <td className="p-1.5 text-right tabular-nums text-zinc-400">
                  {row.speed.toFixed(1)}s
                </td>
                {/* N */}
                <td className="p-1.5 text-right tabular-nums text-zinc-500">
                  {row.n}
                </td>
                {/* Per-scenario */}
                {scenarios.map((sc) => {
                  const v = row.scenarios[sc];
                  const runs = row.scenarioRuns?.[sc] ?? 0;
                  let display: string;
                  let cls: string;
                  if (v != null) {
                    display = String(v);
                    cls = heatClass(v);
                  } else if (runs === 0) {
                    display = "I";
                    cls = "text-zinc-700";
                  } else {
                    display = "\u2014";
                    cls = "text-zinc-600";
                  }
                  return (
                    <td
                      key={sc}
                      className={`p-1.5 text-right tabular-nums ${cls}`}
                    >
                      {display}
                    </td>
                  );
                })}
              </tr>
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
