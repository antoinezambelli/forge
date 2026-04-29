import { useCallback, useEffect, useMemo, useState } from "react";
import type {
  ConfigRow,
  DashboardData,
  FilterDimension,
  Filters,
  ScenarioScope,
  ScreenId,
  SortState,
  SuiteScope,
  ViewId,
} from "./types";
import { FILTER_DIMENSIONS, VIEWS } from "./types";
import { loadData } from "./data";
import { filterByScreen, groupRows, scopeRows } from "./utils";
import { Sidebar } from "./Sidebar";
import { DataTable } from "./DataTable";
import { ComparePanel } from "./ComparePanel";

function initFilters(rows: ConfigRow[]): Filters {
  const filters: Partial<Filters> = {};
  for (const dim of FILTER_DIMENSIONS) {
    filters[dim] = new Set(rows.map((r) => r[dim]));
  }
  return filters as Filters;
}

function App() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [filters, setFilters] = useState<Filters | null>(null);
  const [sort, setSort] = useState<SortState>({ col: "score", asc: false });
  const [checked, setChecked] = useState<number[]>([]);
  const [activeScreen, setActiveScreen] = useState<ScreenId>("reforged");
  const [activeView, setActiveView] = useState<ViewId>("all");
  const [scenarioScope, setScenarioScope] = useState<ScenarioScope>("all");
  const [suiteScope, setSuiteScope] = useState<SuiteScope>("all");

  useEffect(() => {
    loadData().then((d) => {
      setData(d);
      setFilters(initFilters(d.rows));
    });
  }, []);

  const filtered = useMemo(() => {
    if (!data || !filters) return [];
    // Screen drives row set first (which ablations are visible), then
    // the sidebar filter dimensions narrow further.
    const byScreen = filterByScreen(data.rows, activeScreen);
    return byScreen.filter((row) =>
      FILTER_DIMENSIONS.every(
        (dim) => !filters[dim] || filters[dim].has(row[dim]),
      ),
    );
  }, [data, filters, activeScreen]);

  // Apply scenario scope (statefulness) and suite scope — filters scenario
  // columns and recomputes score from the intersected scenario set.
  const { rows: scopedRows, scenarios: scopedScenarios } = useMemo(
    () => scopeRows(
      filtered,
      data?.scenarios ?? [],
      scenarioScope,
      suiteScope,
      data?.scenarioSuite ?? {},
    ),
    [filtered, data, scenarioScope, suiteScope],
  );

  const scopedAbbrev = useMemo(() => {
    if (!data) return {};
    const full = data.scenarioAbbrev;
    const scSet = new Set(scopedScenarios);
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(full)) {
      if (scSet.has(k)) out[k] = v;
    }
    return out;
  }, [data, scopedScenarios]);

  const viewDef = useMemo(
    () => VIEWS.find((v) => v.id === activeView) ?? VIEWS[0],
    [activeView],
  );

  const { sorted, groups } = useMemo(
    () => groupRows(scopedRows, viewDef, sort, scopedScenarios, activeScreen),
    [scopedRows, viewDef, sort, scopedScenarios, activeScreen],
  );

  const totalRuns = useMemo(
    () =>
      filtered.reduce(
        (sum, r) => sum + r.n * (scopedScenarios.length),
        0,
      ),
    [filtered, scopedScenarios],
  );

  const handleFilterChange = useCallback(
    (dim: FilterDimension, val: string, on: boolean) => {
      setFilters((prev) => {
        if (!prev) return prev;
        const next = { ...prev, [dim]: new Set(prev[dim]) };
        if (on) next[dim].add(val);
        else next[dim].delete(val);
        return next;
      });
      setChecked([]);
    },
    [],
  );

  const handleScreenChange = useCallback((id: ScreenId) => {
    setActiveScreen(id);
    setChecked([]);
  }, []);

  const handleViewChange = useCallback((id: ViewId) => {
    setActiveView(id);
    setChecked([]);
  }, []);

  const handleScopeChange = useCallback((scope: ScenarioScope) => {
    setScenarioScope(scope);
    setChecked([]);
  }, []);

  const handleSuiteChange = useCallback((suite: SuiteScope) => {
    setSuiteScope(suite);
    setChecked([]);
  }, []);

  const handleSort = useCallback(
    (col: string) => {
      setSort((prev) =>
        prev.col === col
          ? { col, asc: !prev.asc }
          : { col, asc: col === "label" },
      );
    },
    [],
  );

  const handleCompareToggle = useCallback((idx: number, on: boolean) => {
    setChecked((prev) => {
      if (on) {
        const next = prev.length >= 2 ? [prev[1], idx] : [...prev, idx];
        return next;
      }
      return prev.filter((i) => i !== idx);
    });
  }, []);

  const handleSwap = useCallback(() => {
    setChecked((prev) => [...prev].reverse());
  }, []);

  const handleClear = useCallback(() => {
    setChecked([]);
  }, []);

  if (!data || !filters) {
    return (
      <div className="flex items-center justify-center min-h-screen text-zinc-500">
        Loading...
      </div>
    );
  }

  return (
    <div className="flex min-h-screen">
      <Sidebar
        rows={data.rows}
        filters={filters}
        onFilterChange={handleFilterChange}
        activeScreen={activeScreen}
        onScreenChange={handleScreenChange}
        activeView={activeView}
        onViewChange={handleViewChange}
        scenarioScope={scenarioScope}
        onScopeChange={handleScopeChange}
        suiteScope={suiteScope}
        onSuiteChange={handleSuiteChange}
        filteredCount={filtered.length}
        totalCount={filterByScreen(data.rows, activeScreen).length}
        totalRuns={totalRuns}
        timestamp={data.timestamp}
      />

      <main className="flex-1 min-w-0 p-4 flex flex-col">
        <DataTable
          rows={sorted}
          scenarios={scopedScenarios}
          scenarioAbbrev={scopedAbbrev}
          sort={sort}
          onSort={handleSort}
          checked={checked}
          onCompareToggle={handleCompareToggle}
          groups={groups}
        />

        {checked.length === 2 && (
          <ComparePanel
            a={sorted[checked[0]]}
            b={sorted[checked[1]]}
            scenarios={scopedScenarios}
            scenarioAbbrev={scopedAbbrev}
            onSwap={handleSwap}
            onClear={handleClear}
          />
        )}

        <p className="text-[0.6rem] text-zinc-600 mt-6">
          Generated {data.timestamp}
        </p>
      </main>
    </div>
  );
}

export default App;
