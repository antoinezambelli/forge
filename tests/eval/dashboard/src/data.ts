import type { DashboardData } from "./types";

/**
 * Load dashboard data.
 *
 * In production (built by report.py), data is injected into a global:
 *   <script>window.__FORGE_DATA__ = {...}</script>
 *
 * In development, we fall back to the sample data file.
 */
export async function loadData(): Promise<DashboardData> {
  // Production: report.py injects data as a global
  const global = window as unknown as { __FORGE_DATA__?: DashboardData };
  if (global.__FORGE_DATA__) {
    return global.__FORGE_DATA__;
  }

  // Dev: import sample data
  const mod = await import("./sample-data.json");
  return mod.default as DashboardData;
}
