import type { ConfigRow, FilterDimension, Filters, ScenarioScope, ViewId } from "./types";
import { FILTER_DIMENSIONS, SCENARIO_SCOPES } from "./types";
import { ViewSelector } from "./ViewSelector";

const DIMENSION_LABELS: Record<FilterDimension, string> = {
  backend: "Backend",
  mode: "Mode",
  family: "Family",
  quant: "Quant",
  ablation: "Ablation",
};

interface SidebarProps {
  rows: ConfigRow[];
  filters: Filters;
  onFilterChange: (dim: FilterDimension, val: string, on: boolean) => void;
  activeView: ViewId;
  onViewChange: (id: ViewId) => void;
  scenarioScope: ScenarioScope;
  onScopeChange: (scope: ScenarioScope) => void;
  filteredCount: number;
  totalCount: number;
  totalRuns: number;
  timestamp: string;
}

export function Sidebar({
  rows,
  filters,
  onFilterChange,
  activeView,
  onViewChange,
  scenarioScope,
  onScopeChange,
  filteredCount,
  totalCount,
  totalRuns,
  timestamp,
}: SidebarProps) {
  return (
    <nav className="w-52 min-w-52 shrink-0 border-r border-zinc-800 p-4 sticky top-0 h-screen overflow-y-auto bg-zinc-950/80">
      <h1 className="text-lg font-semibold mb-0.5">Forge Eval</h1>
      <p className="text-xs text-zinc-500 mb-3">
        {filteredCount}/{totalCount} configs &middot;{" "}
        {totalRuns.toLocaleString()} runs
      </p>

      <fieldset className="mb-3 border border-zinc-800 rounded p-2">
        <legend className="text-[0.65rem] font-semibold uppercase tracking-wider text-zinc-400 px-1">
          Scenarios
        </legend>
        <div className="flex flex-wrap gap-1">
          {SCENARIO_SCOPES.map((s) => (
            <button
              key={s.id}
              onClick={() => onScopeChange(s.id)}
              className={`text-[0.65rem] px-2 py-0.5 rounded-full border transition-colors ${
                scenarioScope === s.id
                  ? "border-emerald-500 bg-emerald-500/15 text-emerald-400"
                  : "border-zinc-700 text-zinc-500 hover:border-zinc-500 hover:text-zinc-300"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </fieldset>

      <ViewSelector active={activeView} onChange={onViewChange} />

      {FILTER_DIMENSIONS.map((dim) => {
        const vals = [...new Set(rows.map((r) => r[dim]))].sort();
        if (vals.length < 2) return null;

        return (
          <fieldset
            key={dim}
            className="mb-3 border border-zinc-800 rounded p-2"
          >
            <legend className="text-[0.65rem] font-semibold uppercase tracking-wider text-zinc-400 px-1">
              {DIMENSION_LABELS[dim]}
            </legend>
            {vals.map((v) => (
              <label
                key={v}
                className="flex items-center gap-1.5 text-xs py-0.5 cursor-pointer hover:text-zinc-200"
              >
                <input
                  type="checkbox"
                  checked={filters[dim]?.has(v) ?? true}
                  onChange={(e) => onFilterChange(dim, v, e.target.checked)}
                  className="w-3.5 h-3.5 rounded border-zinc-600 bg-zinc-800 accent-emerald-500"
                />
                <span>{v}</span>
              </label>
            ))}
          </fieldset>
        );
      })}

      <p className="text-[0.6rem] text-zinc-600 mt-4">{timestamp}</p>
    </nav>
  );
}
