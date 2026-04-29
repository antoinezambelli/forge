import type { ConfigRow, ScenarioScope, ScreenId, SortState, SuiteScope, ViewDef } from "./types";
import { ABLATION_ORDER } from "./types";

/** Filter rows according to the active screen.
 *
 *   reforged         — only reforged rows.
 *   bare-vs-reforged — reforged + bare rows (the universally-collected pair).
 *   ablation         — only configs that have more than just reforged+bare;
 *                      i.e. at least one `no_*` ablation exists for that
 *                      (model, backend, mode). Renders the deep-ablation subset.
 */
export function filterByScreen(rows: ConfigRow[], screen: ScreenId): ConfigRow[] {
  if (screen === "reforged") {
    return rows.filter((r) => r.ablation === "reforged");
  }
  if (screen === "bare-vs-reforged") {
    return rows.filter((r) => r.ablation === "reforged" || r.ablation === "bare");
  }
  // ablation: only include configs with at least one no_* variant
  const configHasAblation = new Set<string>();
  for (const r of rows) {
    if (r.ablation.startsWith("no_")) {
      configHasAblation.add(`${r.model}\u0000${r.backend}\u0000${r.mode}`);
    }
  }
  return rows.filter((r) =>
    configHasAblation.has(`${r.model}\u0000${r.backend}\u0000${r.mode}`),
  );
}

/** Rank for sorting ablation rows in canonical order; unknowns land last. */
function ablationRank(name: string): number {
  const idx = ABLATION_ORDER.indexOf(name);
  return idx === -1 ? ABLATION_ORDER.length : idx;
}

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

/** Filter scenarios by scope (statefulness) and suite (og18 / advanced_reasoning),
 * then recompute row aggregates from the filtered scenario set.
 *
 * Both axes intersect: e.g. scope="stateful" + suite="advanced_reasoning"
 * yields the 4 stateful AR scenarios.
 *
 * All shown aggregates (score, accuracy, completeness, efficiency, wasted, speed)
 * are recomputed from per-scenario components emitted by report.py — using the
 * same formulas as the Python side — so a scoped view stays internally
 * consistent. (Older data blobs without per-scenario components fall back to
 * the unscoped values for the metrics they can't reconstruct.)
 */
export function scopeRows(
  rows: ConfigRow[],
  allScenarios: string[],
  scope: ScenarioScope,
  suite: SuiteScope,
  scenarioSuite: Record<string, string>,
): { rows: ConfigRow[]; scenarios: string[] } {
  const isStateful = (sc: string) => sc.endsWith("_stateful");

  let scenarios = allScenarios;
  if (scope === "lambda") {
    scenarios = scenarios.filter((sc) => !isStateful(sc));
  } else if (scope === "stateful") {
    scenarios = scenarios.filter(isStateful);
  }
  if (suite === "og18") {
    scenarios = scenarios.filter((sc) => scenarioSuite[sc] === "og18");
  } else if (suite === "advanced_reasoning") {
    scenarios = scenarios.filter((sc) => scenarioSuite[sc] === "advanced_reasoning");
  }

  // Defensive: if filters wiped out everything, fall back to no-filter view.
  if (scenarios.length === 0) {
    return { rows, scenarios: allScenarios };
  }

  // No filtering applied? Return unmodified.
  if (scenarios.length === allScenarios.length) {
    return { rows, scenarios: allScenarios };
  }

  const recomputed = rows.map((row) => {
    let totalRuns = 0;
    let totalCorrect = 0;
    let totalCompleted = 0;
    let totalValidated = 0;
    let totalIdeal = 0;
    let totalActual = 0;
    let wastedSum = 0;
    let wastedN = 0;
    let speedSum = 0;
    let speedN = 0;

    for (const sc of scenarios) {
      totalRuns      += row.scenarioRuns?.[sc] ?? 0;
      totalCorrect   += row.scenarioCorrect?.[sc] ?? 0;
      totalCompleted += row.scenarioCompleted?.[sc] ?? 0;
      totalValidated += row.scenarioValidated?.[sc] ?? 0;
      totalIdeal     += row.scenarioIdealCalls?.[sc] ?? 0;
      totalActual    += row.scenarioActualCalls?.[sc] ?? 0;
      wastedSum      += row.scenarioWastedSum?.[sc] ?? 0;
      wastedN        += row.scenarioWastedN?.[sc] ?? 0;
      speedSum       += row.scenarioSpeedSum?.[sc] ?? 0;
      speedN         += row.scenarioSpeedN?.[sc] ?? 0;
    }

    // Match Python rounding: percentages to 1 decimal, wasted/speed to 1 decimal.
    const round1 = (x: number) => Math.round(x * 10) / 10;

    const score        = totalRuns > 0       ? round1((totalCorrect / totalRuns) * 100)        : 0;
    const accuracy     = totalValidated > 0  ? round1((totalCorrect / totalValidated) * 100)   : null;
    const completeness = totalRuns > 0       ? round1((totalCompleted / totalRuns) * 100)      : 0;
    const efficiency   = totalActual > 0     ? round1(Math.min(totalIdeal / totalActual, 1) * 100) : 0;
    const wasted       = wastedN > 0         ? round1(wastedSum / wastedN)                     : 0;
    const speed        = speedN > 0          ? round1(speedSum / speedN)                       : 0;

    // Detection: if per-scenario components are missing, fall back to original
    // unscoped values for the metrics we can't reconstruct (older data blobs).
    const hasFullDetail = row.scenarioCompleted !== undefined;

    const n = Math.max(0, ...scenarios.map((sc) => row.scenarioRuns?.[sc] ?? 0));

    return {
      ...row,
      score,
      accuracy:     hasFullDetail ? accuracy     : row.accuracy,
      completeness: hasFullDetail ? completeness : row.completeness,
      efficiency:   hasFullDetail ? efficiency   : row.efficiency,
      wasted:       hasFullDetail ? wasted       : row.wasted,
      speed:        hasFullDetail ? speed        : row.speed,
      n,
    };
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
  screen: ScreenId,
): { sorted: ConfigRow[]; groups: RowGroup[] } {
  // Screens 2 and 3 define their own canonical grouping (per-config ablation
  // towers), regardless of the view picker. The user-selected view only
  // applies on the reforged screen.
  const effectiveView: ViewDef = screen === "reforged"
    ? view
    : {
        id: view.id,
        label: view.label,
        groupBy: ["model", "backend", "mode"],
      };
  const byAblationRank = screen !== "reforged";

  // "All" view with no grouping — just sort flat
  if (effectiveView.groupBy.length === 0) {
    return { sorted: sortRows(rows, sort, scenarios), groups: [] };
  }

  // Bucket rows into groups
  const buckets = new Map<string, ConfigRow[]>();
  for (const row of rows) {
    const k = groupKey(row, effectiveView.groupBy);
    if (!buckets.has(k)) buckets.set(k, []);
    buckets.get(k)!.push(row);
  }

  // Sort within each group.
  //   On ablation-ordered screens, enforce ABLATION_ORDER first.
  //   Otherwise fall back to the view's intraSort.
  const groups: RowGroup[] = [];
  for (const [k, bucket] of buckets) {
    bucket.sort((a, b) => {
      if (byAblationRank) {
        const diff = ablationRank(a.ablation) - ablationRank(b.ablation);
        if (diff !== 0) return diff;
        return b.score - a.score;
      }
      const scoreDiff = b.score - a.score;
      if (scoreDiff !== 0) return scoreDiff;
      if (effectiveView.intraSort) {
        return String(a[effectiveView.intraSort]).localeCompare(
          String(b[effectiveView.intraSort]),
        );
      }
      return 0;
    });

    // Derive a human-readable group label from the groupBy fields
    const label = effectiveView.groupBy.map((f) => bucket[0][f]).join(" / ");
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
