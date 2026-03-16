# Forge — Public Release Polish Plan

Findings and recommendations for making the repo audience-ready.

---

## 1. README.md — First Impressions

- **Add a one-liner elevator pitch at the very top** — badges + "Define tools, pick a backend, run agentic workflows locally."
- **Add a "Quick Start" section** before the full usage example — 5 lines, one tool, one call. Current example is ~40 lines and introduces `setup_backend`, `BudgetMode`, `ContextManager` all at once.
- **Move Multi-Turn to a separate doc or collapse it** — README is already long. Host the full multi-turn example in `docs/WORKFLOW.md` or a new `docs/USAGE.md`, link from README.
- **Add a "Which backend should I use?" mini-table:**

  | Backend | Best for | Native FC? |
  |---------|----------|------------|
  | Ollama | Easiest setup, model management built-in | Yes |
  | llama-server | Best performance, full control | Yes (with `--jinja`) |
  | Llamafile | Single binary, zero deps | No (prompt-injected) |
  | Anthropic | Frontier baseline / hybrid | Yes |

- **Simplify Roadmap items 3 and 4** — currently implementation-level details (compaction thresholds, self-awareness). Rewrite in user-facing language or move to `CONTRIBUTING.md`.

---

## 2. pyproject.toml — Packaging Metadata

- **Add missing metadata:** `authors`, `readme`, `urls` (homepage, repository, docs).
  ```toml
  authors = [{name = "Antoine Zambelli"}]
  readme = "README.md"
  [project.urls]
  Repository = "https://github.com/antoinezambelli/forge"
  Documentation = "https://github.com/antoinezambelli/forge/tree/main/docs"
  ```
- **Split `anthropic` into its own optional extra** (`[anthropic]`) — currently bundled with dev deps, so users who want `AnthropicClient` must also pull pytest.
- **Add `[project.classifiers]`** — Python version classifiers, topic tags for discoverability.

---

## 3. Results Docs — Highest Impact Opportunity

10 markdown files of raw ASCII tables with zero narrative. 103 configs, no recommendations.

- **Create `docs/results/RECOMMENDATIONS.md`** — human-readable guide organized by use case:
  - "I have 8GB VRAM" → Ministral-8B instruct on Ollama (94.6%) or llama-server native (97.4%)
  - "I have 12–16GB VRAM" → Ministral-14B reasoning on llama-server (99.3%) or Qwen3-14B on Ollama (96.3%)
  - "I want maximum reliability" → Claude Sonnet via Anthropic (100%) or Ministral-8B reasoning (99.3%)
  - "I want cheapest/fastest local" → Ministral-8B instruct, Ollama, ~94–97%
  - "I'm on Mac / CPU-only" → guidance needed or "not tested yet"

- **Add a "Key Findings" section to `results/index.md`** — 5–6 bullets:
  - Forge guardrails add 10–55% accuracy depending on model
  - Native FC vs prompt-injected: ~1–2% gap (smaller than expected)
  - Ollama vs llama-server: ~2–4% gap on same model
  - Ministral family is the sweet spot for local deployment
  - Llama 3.1 and Mistral v0.3 not recommended (60–65%, 0–65%)

- **Trim/reorganize views** — `bare-vs-reforged.md` (64.9KB), `all.md`, `by-backend.md` overlap heavily. Audience-facing views: `RECOMMENDATIONS.md`, `by-family.md`, `ablation.md`. Move the rest to `results/raw/` for power users.

---

## 4. docs/BACKEND_SETUP.md — Practical Gaps

- References `ref_docs/Modelfile.ministral` which doesn't exist in the repo (forge-dev artifact).
- **Add a VRAM requirements table:** model name → quantization → VRAM needed → recommended backend.
- **Add a "Verify your setup" section** — a single smoke-test command users can run after install.

---

## 5. docs/ARCHITECTURE.md — Audience Targeting

- At 77KB this is a comprehensive internal design doc — great for contributors, wrong first doc for users.
- **Consider adding `docs/GETTING_STARTED.md`** (or expanding README): install → pick backend → define workflow → run → observe results. Link to ARCHITECTURE.md for the deep dive.

---

## 6. Minor Polish

- **`__init__.py` — `ToolParam` missing from `__all__`** — README usage example imports it but it's not exported at the top level.
- **No LICENSE file** — `pyproject.toml` claims Apache-2.0 but no `LICENSE` or `LICENSE.txt` in repo root.
- **No CONTRIBUTING.md** — even a minimal one helps set expectations for a public repo.
- **Dashboard location** — `tests/eval/dashboard/` has its own `package.json` inside `tests/`. Note in README that it's a separate build step, or consider moving to top-level `dashboard/`.

---

## Priority Order

1. **LICENSE file** — blocking for any public repo with Apache-2.0 claim
2. **pyproject.toml metadata** — authors, URLs, readme pointer
3. **Results recommendations doc** — highest value-add for users choosing a config
4. **README quick-start + backend table** — first impression improvements
5. **BACKEND_SETUP VRAM table + smoke test** — practical onboarding
6. **`__init__.py` ToolParam export** — small consistency fix
7. **Trim/reorganize results** — nice-to-have for clarity
