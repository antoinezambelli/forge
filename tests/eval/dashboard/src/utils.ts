import type { ConfigRow, ScenarioScope, SortState, ViewDef } from "./types";

/** Heat-map color class based on percentage value. */
export function heatClass(v: number | null): string {
  if (v == null) return "";
  if (v >= 95) return "text-emerald-400";
  if (v >= 90) return "text-emerald-500/80";
  if (v >= 70) return "text-amber-400";
  if (v >= 50) return "text-orange-400";
  return "text-red-400";
}

/** Format a percentage value for display. */
export function fmtPct(v: number | null, decimals: number = 0): string {
  if (v == null) return "\u2014";
  return `${v.toFixed(decimals)}%`;
}

/** Filter scenarios by scope and recompute row aggregates.
 *
 * When scope is "lambda" or "stateful", we filter the scenario list and
 * recompute score from the per-scenario values.
 * Other aggregates (completeness, efficiency, wasted, speed) are kept
 * as-is since the data blob doesn't carry per-scenario breakdowns for those.
 */
export function scopeRows(
  rows: ConfigRow[],
  allScenarios: string[],
  scope: ScenarioScope,
): { rows: ConfigRow[]; scenarios: string[] } {
  if (scope === "all") {
    return { rows, scenarios: allScenarios };
  }

  const isStateful = (sc: string) => sc.endsWith("_stateful");
  const scenarios = scope === "stateful"
    ? allScenarios.filter(isStateful)
    : allScenarios.filter((sc) => !isStateful(sc));

  if (scenarios.length === 0) {
    return { rows, scenarios: allScenarios };
  }

  const recomputed = rows.map((row) => {
    // Recompute using the same formula as Python: score = total_correct / total_runs
    let totalRuns = 0;
    let totalCorrect = 0;
    for (const sc of scenarios) {
      const runs = row.scenarioRuns?.[sc] ?? 0;
      const correct = row.scenarioCorrect?.[sc] ?? 0;
      totalRuns += runs;
      totalCorrect += correct;
    }
    const score = totalRuns > 0
      ? Math.round(totalCorrect / totalRuns * 1000) / 10
      : 0;

    // Recompute N as max run count across scoped scenarios
    const n = Math.max(0, ...scenarios.map((sc) => row.scenarioRuns?.[sc] ?? 0));

    return { ...row, score, n };
  });

  return { rows: recomputed, scenarios };
}

/** Sort comparator for the data table. */
export function sortRows(
  rows: ConfigRow[],
  sort: SortState,
  scenarios: string[],
): ConfigRow[] {
  const sorted = [...rows];
  sorted.sort((a, b) => {
    let va: number | string;
    let vb: number | string;

    if (sort.col === "label") {
      va = a.label;
      vb = b.label;
    } else if (scenarios.includes(sort.col)) {
      va = a.scenarios[sort.col] ?? -1;
      vb = b.scenarios[sort.col] ?? -1;
    } else {
      va = (a as unknown as Record<string, number>)[sort.col] ?? -1;
      vb = (b as unknown as Record<string, number>)[sort.col] ?? -1;
    }

    if (typeof va === "string" && typeof vb === "string") {
      return sort.asc ? va.localeCompare(vb) : vb.localeCompare(va);
    }
    return sort.asc
      ? (va as number) - (vb as number)
      : (vb as number) - (va as number);
  });
  return sorted;
}

/** Build a group key string from a row using the view's groupBy fields. */
function groupKey(row: ConfigRow, fields: (keyof ConfigRow)[]): string {
  return fields.map((f) => String(row[f])).join("\u0000");
}

export interface RowGroup {
  key: string;
  label: string;
  rows: ConfigRow[];
}

/**
 * Group and sort rows according to a view definition.
 * - Groups rows by the view's groupBy fields
 * - Sorts within groups by intraSort (asc) then score (desc)
 * - Sorts groups by best score descending
 * Returns flat row array + group boundary indices for separator rendering.
 */
export function groupRows(
  rows: ConfigRow[],
  view: ViewDef,
  sort: SortState,
  scenarios: string[],
): { sorted: ConfigRow[]; groups: RowGroup[] } {
  // "All" view — no grouping, just normal sort
  if (view.groupBy.length === 0) {
    return { sorted: sortRows(rows, sort, scenarios), groups: [] };
  }

  // Bucket rows into groups
  const buckets = new Map<string, ConfigRow[]>();
  for (const row of rows) {
    const k = groupKey(row, view.groupBy);
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k)!.push(row);
  }

  // Sort within each group: by intraSort field (asc) then by score (desc)
  const groups: RowGroup[] = [];
  for (const [k, bucket] of buckets) {
    bucket.sort((a, b) => {
      const scoreDiff = b.score - a.score;
      if (scoreDiff !== 0) return scoreDiff;
      if (view.intraSort) {
        return String(a[view.intraSort]).localeCompare(String(b[view.intraSort]));
      }
      return 0;
    });

    // Derive a human-readable group label from the groupBy fields
    const label = view.groupBy.map((f) => bucket[0][f]).join(" / ");
    groups.push({ key: k, label, rows: bucket });
  }

  // Sort groups by best score in each group (descending)
  groups.sort((a, b) => {
    const bestA = Math.max(...a.rows.map((r) => r.score));
    const bestB = Math.max(...b.rows.map((r) => r.score));
    return bestB - bestA;
  });

  // Flatten
  const sorted = groups.flatMap((g) => g.rows);
  return { sorted, groups };
}
