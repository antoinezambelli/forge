import { describe, expect, it } from "vitest";
import type { ConfigRow } from "./types";
import { scopeRows } from "./utils";

const LAMBDA = "basic_2step";
const STATEFUL = "basic_2step_stateful";

function row(): ConfigRow {
  return {
    label: "test",
    model: "model-q4",
    backend: "llamaserver",
    mode: "native",
    ablation: "reforged",
    replay: "none",
    family: "model",
    quant: "q4",
    gen: 1,
    retired: false,
    score: 50,
    validatedAccuracy: 75,
    completionRate: 50,
    attemptedCount: 6,
    correctCount: 3,
    validatedCount: 4,
    completedCount: 3,
    efficiency: 100,
    wasted: 0,
    speed: 1,
    n: 4,
    scenarios: { [LAMBDA]: 25, [STATEFUL]: 100 },
    scenarioAttempted: { [LAMBDA]: 4, [STATEFUL]: 2 },
    scenarioCorrect: { [LAMBDA]: 1, [STATEFUL]: 2 },
    scenarioValidated: { [LAMBDA]: 2, [STATEFUL]: 2 },
    scenarioCompleted: { [LAMBDA]: 2, [STATEFUL]: 1 },
    scenarioIdealCalls: { [LAMBDA]: 2, [STATEFUL]: 4 },
    scenarioActualCalls: { [LAMBDA]: 4, [STATEFUL]: 4 },
    scenarioWastedSum: { [LAMBDA]: 2, [STATEFUL]: 0 },
    scenarioWastedN: { [LAMBDA]: 2, [STATEFUL]: 1 },
    scenarioSpeedSum: { [LAMBDA]: 4, [STATEFUL]: 2 },
    scenarioSpeedN: { [LAMBDA]: 2, [STATEFUL]: 1 },
  };
}

describe("scopeRows", () => {
  it("recomputes score, validated accuracy, and completion independently", () => {
    const result = scopeRows(
      [row()],
      [LAMBDA, STATEFUL],
      "lambda",
      "all",
      { [LAMBDA]: "og18", [STATEFUL]: "og18" },
    );

    expect(result.scenarios).toEqual([LAMBDA]);
    expect(result.rows[0]).toMatchObject({
      score: 25,
      validatedAccuracy: 50,
      completionRate: 50,
      attemptedCount: 4,
      correctCount: 1,
      validatedCount: 2,
      completedCount: 2,
      n: 4,
    });
  });
});
