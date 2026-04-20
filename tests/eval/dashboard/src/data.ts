import type { DashboardData } from "./types";

/**
 * Dashboard data is injected by report.py into a global before React boots:
 *   <script>window.__FORGE_DATA__ = {...}</script>
 */
export async function loadData(): Promise<DashboardData> {
  const global = window as unknown as { __FORGE_DATA__?: DashboardData };
  if (!global.__FORGE_DATA__) {
    throw new Error(
      "window.__FORGE_DATA__ not injected — build via `python -m tests.eval.report <jsonl> --html <out>`",
    );
  }
  return global.__FORGE_DATA__;
}
