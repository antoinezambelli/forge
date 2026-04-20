/** A single config row as produced by report.py _metrics_to_json_row(). */
export interface ConfigRow {
  label: string;
  model: string;
  backend: string;
  mode: string;
  ablation: string;
  family: string;
  quant: string;
  score: number;
  accuracy: number | null;
  completeness: number;
  efficiency: number;
  wasted: number;
  speed: number;
  n: number;
  scenarios: Record<string, number | null>;
  scenarioRuns?: Record<string, number>;
  scenarioCorrect?: Record<string, number>;
}

/** The full data blob injected by report.py into the built HTML. */
export interface DashboardData {
  rows: ConfigRow[];
  scenarios: string[];
  scenarioAbbrev: Record<string, string>;
  timestamp: string;
}

export interface SortState {
  col: string;
  asc: boolean;
}

/** Scenario scope — controls which scenarios are included in aggregates and columns. */
export type ScenarioScope = "all" | "lambda" | "stateful";

export const SCENARIO_SCOPES: { id: ScenarioScope; label: string }[] = [
  { id: "all", label: "All" },
  { id: "lambda", label: "Lambda" },
  { id: "stateful", label: "Stateful" },
];

export const FILTER_DIMENSIONS = ["backend", "mode", "family", "quant"] as const;
export type FilterDimension = (typeof FILTER_DIMENSIONS)[number];

export type Filters = Record<FilterDimension, Set<string>>;

/** Screen — top-level mode selector that controls which ablation rows are visible.
 *
 *   reforged         — one row per (model, backend, mode): the "pick a model" leaderboard.
 *   bare-vs-reforged — reforged + bare, grouped: the "how much does forge lift this model?" story.
 *   ablation         — all ablation variants per config, grouped: the per-guardrail contribution deep dive.
 */
export type ScreenId = "reforged" | "bare-vs-reforged" | "ablation";

export const SCREENS: { id: ScreenId; label: string }[] = [
  { id: "reforged", label: "Reforged" },
  { id: "bare-vs-reforged", label: "Reforged vs Bare" },
  { id: "ablation", label: "Full Ablation" },
];

/** Intra-group ordering for ablation rows. Defines the canonical ablation sequence. */
export const ABLATION_ORDER: readonly string[] = [
  "reforged",
  "bare",
  "no_rescue",
  "no_nudge",
  "no_steps",
  "no_recovery",
  "no_compact",
];

/** Pre-baked view definitions — control grouping within the active screen's row set. */
export type ViewId = "all" | "by-backend" | "by-family";

export interface ViewDef {
  id: ViewId;
  label: string;
  /** Fields that form the group key (rows with same key are grouped together). */
  groupBy: (keyof ConfigRow)[];
  /** Sort rows within each group by this field (descending by default). */
  intraSort?: keyof ConfigRow;
}

export const VIEWS: ViewDef[] = [
  { id: "all", label: "All", groupBy: [] },
  {
    id: "by-backend",
    label: "By Backend",
    groupBy: ["model", "quant", "ablation"],
    intraSort: "backend",
  },
  {
    id: "by-family",
    label: "By Family",
    groupBy: ["family"],
  },
];
