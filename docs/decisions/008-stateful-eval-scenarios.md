# ADR-008: Stateful Eval Scenarios

**Status:** Done

## Problem

All 11 eval scenarios use argument-blind tool callables. `get_info(query="rome")`
returns the Paris canned string. `check_supplier(supplier="anything")` routes
through a fuzzy match but fundamentally returns a static blob. The only exception
is `error_recovery`, which validates a 4-digit format — a type check, not
stateful behavior.

This tests "can the model navigate a workflow" (correct tool sequence, step
enforcement, compaction survival, error recovery) but NOT "can the model use
tools correctly in a stateful environment." A model that passes `entity_id="999"`
instead of `entity_id="42"` gets the same result from `_fetch_details` — the
wrong-argument path returns a "not recognized" string that the model can still
parrot into the terminal tool and pass validation.

The gap: **tool results that depend on arguments and prior state.** If the model
calls `mv(source="wrong.txt", destination="temp")` on a stateful filesystem,
the file doesn't move. If it then calls `grep(file_name="wrong.txt")`, grep
fails because the file is in the wrong place. Mistakes compound. This is the
BFCL pattern, and it's what forge's eval suite lacks.

## Decision

Add 9 stateful scenarios that mirror the existing non-compaction scenarios.
Each gets its own backend class with methods wired as `ToolDef` callables.
Validation checks the backend's end state, not just terminal tool arguments.

### What changes

**1. `EvalScenario` — add `validate_state` field**

```python
@dataclass
class EvalScenario:
    ...
    validate: Callable[[dict[str, Any]], bool] | None = None       # existing: terminal args
    validate_state: Callable[[], bool] | None = None               # new: backend end state
```

Closures over the backend instance. Called after the workflow completes. Both
fields are optional and independent — a scenario can have one or both.

**2. `eval_runner.py` — check `validate_state` after workflow**

```python
# After workflow completes, existing terminal-args check:
if scenario.validate and capture.get("args") is not None:
    accuracy = scenario.validate(capture["args"])

# New state check (additive):
if scenario.validate_state is not None:
    state_ok = scenario.validate_state()
    # Combine: both must pass if both present
    if accuracy is None:
        accuracy = state_ok
    else:
        accuracy = accuracy and state_ok
```

**3. `scenarios.py` — convert existing scenarios from `make_tool` to `ToolDef`**

All 11 existing scenarios switch from `make_tool()` to direct `ToolDef`/`ToolSpec`/
`ToolParam` construction. This mirrors the prod API surface. `make_tool` is
deleted (or left as dead code for one release, then deleted).

Example conversion:

```python
# Before (make_tool — all params type "string", hides ToolSpec)
"get_info": make_tool(
    "get_info",
    "Retrieve information about a topic.",
    {"query": "The topic to look up"},
    lambda **kwargs: "The capital of France is Paris. Population: 2.1 million.",
),

# After (direct ToolDef — explicit types, matches prod usage)
"get_info": ToolDef(
    spec=ToolSpec(
        name="get_info",
        description="Retrieve information about a topic.",
        parameters={
            "query": ToolParam(type="string", description="The topic to look up"),
        },
    ),
    callable=lambda **kwargs: "The capital of France is Paris. Population: 2.1 million.",
),
```

**4. `scenarios.py` (or `stateful_scenarios.py`) — 9 new stateful scenarios**

Each scenario has:
- A backend class with stateful methods
- Tools wired as `ToolDef(callable=backend.method_name)`
- `validate_state` closure that checks the backend's end state
- Same tags, budget, iteration limits as its lambda counterpart (where applicable)
- A `"stateful"` tag for filtering

**5. `_build_workflow_with_capture` — backend factory per run**

Stateful scenarios need a fresh backend instance per eval run. The current
`_build_workflow_with_capture` copies the workflow's tool dict and wraps the
terminal callable. For stateful scenarios, the scenario must provide a factory
that creates a fresh backend + fresh tools + fresh `validate_state` closure.

Approach: add an optional `build_workflow` factory to `EvalScenario`:

```python
@dataclass
class EvalScenario:
    ...
    build_workflow: Callable[[], tuple[Workflow, Callable[[], bool] | None]] | None = None
```

If present, `eval_runner` calls it instead of copying the scenario's static
workflow. The factory returns `(workflow, validate_state_fn)`. This keeps
backend instantiation and wiring colocated with the scenario definition.

Lambda scenarios don't set this field — they use the existing static workflow
path unchanged.

### The 9 stateful scenarios

Each mirrors an existing lambda scenario's *workflow shape* (same number of
required steps, same tool count, similar system prompt) but with a stateful
backend where arguments matter and state carries forward.

| # | Lambda original | Stateful version | Backend concept | Key stateful behavior |
|---|---|---|---|---|
| 1 | `basic_2step` | `basic_2step_stateful` | Country facts DB | `get_country_info(country)` returns facts only for known countries; `summarize(content)` must reference actual retrieved value |
| 2 | `sequential_3step` | `sequential_3step_stateful` | Data pipeline | `fetch(source)` populates internal dataset; `analyze(data)` computes stats from actual data; `report(findings)` must cite computed values |
| 3 | `error_recovery` | `error_recovery_stateful` | Inventory system | `fetch(count)` validates format AND checks stock levels; model must retry with valid count that's actually in stock |
| 4 | `tool_selection` | `tool_selection_stateful` | User/permissions DB | `lookup_user(name)` returns different data per name; `get_permissions(user)` returns different perms per user; distractors present |
| 5 | `argument_fidelity` | `argument_fidelity_stateful` | Entity registry | `lookup(entity)` returns a specific ID; `fetch_details(entity_id)` returns data only for valid IDs; wrong ID = empty/error |
| 6 | `sequential_reasoning` | `sequential_reasoning_stateful` | Medical records DB | `get_records(patient_id)` returns different meds per patient; `check_interactions(medication)` returns interactions specific to that med |
| 7 | `conditional_routing` | `conditional_routing_stateful` | Monitoring system | `check_metrics(service)` returns time-series data; `check_deployment(service)` returns deploy info; timestamps must be correlated by model |
| 8 | `data_gap_recovery` | `data_gap_recovery_stateful` | HR records system | `get_employee(name)` returns partial record with hints; `security_audit(id)` / `onboarding_records(id)` return data only for valid IDs |
| 9 | `relevance_detection` | `relevance_detection_stateful` | Travel booking system | Tools actually track state (bookings made, currencies converted); model should still call `decline` — state should remain untouched |

### Design philosophy: tool descriptions as realistic API docs

Tool names, descriptions, and parameter descriptions should read like a
developer documented their API — scoped to what the endpoint actually does,
not generic placeholders. The model gets: system prompt + tool JSON schemas +
user message. No class docstrings, no source code, no enumeration of valid
values in the schema (unless a real API would do so, e.g. an `enum` field).

**Guiding principles:**

- **Name tools after what they do.** `get_country_info`, not `get_info`.
  `check_stock`, not `fetch`. A developer with a country-facts endpoint
  would name it accordingly.
- **Describe params in domain terms.** `"Country name"`, not `"The topic
  to look up"`. Enough context for the model to derive the right argument
  from the user message without hand-holding.
- **Fuzzy-match where a real API would be forgiving.** Case-insensitive
  matching, trimming whitespace — the kind of normalization a real backend
  does. But `get_country_info(country="capital of France")` should fail;
  "capital of France" is not a country name.
- **Don't fish for perfect scores.** The goal is realistic tool use, not
  gotcha string matching. If a competent developer reading the tool schema
  and user message would know what to pass, the model should too.
- **Wrong arguments produce wrong results, not crashes.** A bad key returns
  "No entry found for 'xyz'", not an exception. The model can still
  recover if the workflow allows retries, but the wrong data propagates
  downstream and fails validation.

**Backend principles**: Each backend should be 30-80 lines. Simple enough to
understand at a glance, complex enough that wrong arguments produce wrong
results. No external dependencies. All state is instance attributes.

**Validation**: `validate_state()` checks specific end-state conditions:
- Did the right records get created/modified?
- Are internal data structures in the expected state?
- Did the model NOT mutate state it shouldn't have? (relevance_detection)

### Naming

Stateful scenarios use `[original_name]_stateful` suffix to distinguish them
in reports. Tags include both `"stateful"` and the original scenario's tags
(e.g., `["stateful", "model_quality"]`).

### Scenario details

#### 1. `basic_2step_stateful` — Country Facts DB

Mirrors `basic_2step`. 2-step workflow: look up, then summarize.

**Backend class: `CountryFactsDB`**

```python
class CountryFactsDB:
    def __init__(self):
        self.data = {
            "france": "Capital: Paris. Population: 2.1 million (city), 67 million (country).",
            "japan": "Capital: Tokyo. Population: 14 million (city), 125 million (country).",
        }
        self.last_retrieved = None

    def get_country_info(self, country: str) -> str:
        key = country.strip().lower()
        if key in self.data:
            self.last_retrieved = self.data[key]
            return self.data[key]
        return f"No entry found for '{country}'."

    def summarize(self, content: str) -> str:
        return content  # echo-back terminal
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `get_country_info` | Look up facts about a country. | `country` (string): Country name |
| `summarize` | Summarize content and provide the final answer. | `content` (string): The content to summarize |

**System prompt:** "You are a helpful assistant. Use the available tools to
answer the user's question. First use get_country_info to retrieve
information, then use summarize to provide the final answer."

**User message:** "What is the capital of France?"

**Validation:**
- `validate` (terminal args): `content` mentions "paris" and "capital"
- `validate_state`: `last_retrieved is not None` (a valid entry was found)

**Tags:** `["stateful", "plumbing"]` | **Ideal iterations:** 2

#### 2. `sequential_3step_stateful` — Sales Data Pipeline

Mirrors `sequential_3step`. 3-step workflow: fetch → analyze → report.
Fetch arg matters (quarter + year select the dataset). Analyze reads from
loaded state and computes stats. Report must cite the computed values.

**Backend class: `SalesPipeline`**

```python
class SalesPipeline:
    def __init__(self):
        self.datasets = {
            (4, 2024): {
                "records": 150, "columns": 12,
                "revenue": 11_564_600, "yoy_growth": 23.1,
                "top_product": "Widget Pro", "weakest_region": "APAC",
            },
            (3, 2024): {
                "records": 130, "columns": 12,
                "revenue": 9_400_000, "yoy_growth": 18.5,
                "top_product": "Widget Standard", "weakest_region": "EMEA",
            },
        }
        self.loaded_data = None
        self.analysis = None

    def fetch_sales_data(self, quarter: int, year: int) -> str:
        key = (quarter, year)
        if key in self.datasets:
            self.loaded_data = self.datasets[key]
            d = self.loaded_data
            return (
                f"Loaded Q{quarter} {year} sales data: "
                f"{d['records']} records, {d['columns']} columns."
            )
        return f"No dataset found for Q{quarter} {year}."

    def analyze_sales(self) -> str:
        if self.loaded_data is None:
            return "Error: no data loaded. Call fetch_sales_data first."
        d = self.loaded_data
        self.analysis = (
            f"Revenue: ${d['revenue']:,}. "
            f"YoY growth: {d['yoy_growth']}%. "
            f"Top product: {d['top_product']}. "
            f"Weakest region: {d['weakest_region']}."
        )
        return self.analysis

    def report(self, findings: str) -> str:
        return findings  # echo-back terminal
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `fetch_sales_data` | Fetch quarterly sales data. | `quarter` (integer): Quarter number (1-4); `year` (integer): Four-digit year |
| `analyze_sales` | Analyze the loaded sales dataset and produce findings. | *(none)* |
| `report` | Produce a final report from findings. | `findings` (string): The findings to include in the report |

**System prompt:** "You are a data analyst assistant. Fetch the sales data
first, then analyze it, then produce a report using the report tool."

**User message:** "Generate a sales report from the Q4 2024 dataset."

**Validation:**
- `validate` (terminal args): `findings` mentions "23" (growth), "widget pro",
  and "apac"
- `validate_state`: `loaded_data` is the Q4 2024 dataset AND `analysis` is
  not None (both steps executed against the correct data)

**What's tested beyond `sequential_3step`:**
- `fetch_sales_data` takes typed integer params — model must extract quarter
  and year from user message and pass them as integers, not strings
- Wrong quarter/year → wrong dataset → wrong computed stats → validation fails
- `analyze_sales` has no params — state carries implicitly from fetch
- Report must cite *computed* values (from analyze), not raw values (from fetch)

**Tags:** `["stateful", "plumbing"]` | **Ideal iterations:** 3

#### 3. `error_recovery_stateful` — Inventory System

Mirrors `error_recovery`. Same zero-padded 4-digit string trip — model will
pass `"10"`, get a TypeError, retry with `"0010"`, succeed. The stateful
addition: success actually mutates inventory state.

**Backend class: `InventorySystem`**

```python
class InventorySystem:
    def __init__(self):
        self.stock = {"WP-1001": 150}
        self.reservations = []

    def reserve_stock(self, sku: str, quantity: str) -> str:
        if not (isinstance(quantity, str) and quantity.isdigit()
                and len(quantity) == 4):
            raise TypeError(
                f"quantity must be a zero-padded 4-digit string, got '{quantity}'"
            )
        if sku not in self.stock:
            return f"Unknown SKU '{sku}'."
        amt = int(quantity)
        if amt > self.stock[sku]:
            return f"Insufficient stock for {sku}. Available: {self.stock[sku]}."
        self.stock[sku] -= amt
        self.reservations.append({"sku": sku, "quantity": amt})
        return f"Reserved {amt} units of {sku}. Remaining: {self.stock[sku]}."

    def confirm_reservation(self, summary: str) -> str:
        return summary  # echo-back terminal
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `reserve_stock` | Reserve units of a product from inventory. | `sku` (string): Product SKU; `quantity` (string): Number of units as a zero-padded 4-digit string |
| `confirm_reservation` | Confirm and summarize the reservation. | `summary` (string): Reservation summary |

**System prompt:** "You are a helpful assistant. Reserve the requested stock,
then confirm the reservation."

**User message:** "Reserve 10 units of WP-1001."

**Validation:**
- `validate` (terminal args): `summary` mentions "10" and "WP-1001"
- `validate_state`: `stock["WP-1001"] == 140` AND `len(reservations) == 1`

**What's tested beyond `error_recovery`:**
- Same format trip as original (zero-padded 4-digit string)
- On success, stock actually decrements and reservation is recorded
- State validation confirms the mutation happened exactly once (no
  double-reserve from multiple retries before success)

**Tags:** `["stateful", "plumbing"]` | **Ideal iterations:** 3 (fail, retry, confirm)

#### 4. `tool_selection_stateful` — User/Permissions DB

Mirrors `tool_selection`. Same crowded 8-tool namespace (6 distractors + 2
relevant + 1 terminal). The stateful addition: `lookup_user` returns a
`user_id` that must be threaded into `get_permissions`. Wrong name → wrong
ID → wrong permissions → validation fails. Mistakes cascade silently.

**Backend class: `UserPermissionsDB`**

```python
class UserPermissionsDB:
    def __init__(self):
        self.users = {
            "alice": {"role": "Engineer", "team": "Platform", "user_id": "U-1001"},
            "bob": {"role": "Manager", "team": "Infrastructure", "user_id": "U-1002"},
        }
        self.permissions = {
            "U-1001": ["read", "write", "admin"],
            "U-1002": ["read"],
        }
        self.looked_up = None
        self.perms_fetched = None

    def lookup_user(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.users:
            u = self.users[key]
            self.looked_up = u
            return (
                f"User: {name.title()}, Role: {u['role']}, "
                f"Team: {u['team']}, ID: {u['user_id']}"
            )
        return f"No user found for '{name}'."

    def get_permissions(self, user_id: str) -> str:
        uid = user_id.strip()
        if uid in self.permissions:
            perms = self.permissions[uid]
            self.perms_fetched = perms
            return f"Permissions for {uid}: {', '.join(perms)}"
        return f"No permissions found for '{user_id}'."

    def respond(self, answer: str) -> str:
        return answer  # echo-back terminal
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `lookup_user` | Look up a user by name. | `name` (string): The user's name |
| `get_permissions` | Get permissions for a user by their user ID. | `user_id` (string): The user's ID |
| `respond` | Provide the final answer to the user. | `answer` (string): The final answer |
| `search_web` | Search the web for information. | `query` (string): The search query |
| `read_file` | Read a file from disk. | `path` (string): Path to the file |
| `list_directory` | List contents of a directory. | `path` (string): Path to the directory |
| `run_command` | Run a shell command. | `cmd` (string): The command to run |
| `send_email` | Send an email. | `to` (string): Recipient email address |

*(6 distractors return canned strings — same as original lambda scenario.)*

**System prompt:** "You are an admin assistant. Use the available tools to
answer the user's question. Look up the user first, then check their
permissions, then respond."

**User message:** "What permissions does Alice have?"

**Validation:**
- `validate` (terminal args): `answer` contains "read", "write", "admin"
- `validate_state`: `looked_up is not None` AND
  `perms_fetched == ["read", "write", "admin"]`

**What's tested beyond `tool_selection`:**
- Model must extract `U-1001` from `lookup_user`'s response and pass it to
  `get_permissions` — the ID-threading chain. Passing `name="alice"` to
  `get_permissions` returns "No permissions found" (param is `user_id`,
  not `name`).
- Same distractor challenge as original — 6 irrelevant tools in the namespace
- State validation confirms the right user was looked up AND the right
  permissions were fetched (not fabricated by the model)

**Tags:** `["stateful", "model_quality"]` | **Ideal iterations:** 3

#### 5. `argument_fidelity_stateful` — Entity Registry

Mirrors `argument_fidelity`. Same 3-tool shape (lookup → fetch → present),
no distractors. The stateful addition: realistic alphanumeric entity IDs
(`ENT-4728` vs the original's bare `42`) and a decoy entity that produces
different data if the model looks up the wrong name.

**Backend class: `EntityRegistry`**

```python
class EntityRegistry:
    def __init__(self):
        self.entities = {
            "widget pro": {
                "entity_id": "ENT-4728", "status": "active",
                "owner": "alice@example.com",
            },
            "widget basic": {
                "entity_id": "ENT-3301", "status": "retired",
                "owner": "bob@example.com",
            },
        }
        self.details = {
            "ENT-4728": {
                "name": "Widget Pro", "created": "2024-01-15",
                "units_sold": 1500, "category": "Premium",
            },
            "ENT-3301": {
                "name": "Widget Basic", "created": "2022-06-01",
                "units_sold": 800, "category": "Standard",
            },
        }
        self.looked_up = None
        self.fetched = None

    def lookup_entity(self, entity: str) -> str:
        key = entity.strip().lower()
        if key in self.entities:
            e = self.entities[key]
            self.looked_up = e
            return (
                f"Entity ID: {e['entity_id']}, "
                f"Status: {e['status']}, Owner: {e['owner']}"
            )
        return f"No entity found for '{entity}'."

    def fetch_details(self, entity_id: str) -> str:
        eid = entity_id.strip()
        if eid in self.details:
            d = self.details[eid]
            self.fetched = d
            return (
                f"Details: {d['name']}, created {d['created']}, "
                f"{d['units_sold']} units sold, category: {d['category']}"
            )
        return f"No details found for entity ID '{entity_id}'."

    def present(self, summary: str) -> str:
        return summary  # echo-back terminal
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `lookup_entity` | Look up an entity by name. | `entity` (string): Entity name |
| `fetch_details` | Fetch details for an entity by its ID. | `entity_id` (string): The entity's ID |
| `present` | Present the final summary to the user. | `summary` (string): The summary to present |

**System prompt:** "You are a helpful assistant. Look up the entity, then
fetch its details using the entity ID from the lookup result, then present
a summary."

**User message:** "Look up the entity 'Widget Pro' and get its details."

**Validation:**
- `validate` (terminal args): `summary` mentions "widget pro" and "1500"
- `validate_state`: `looked_up is not None` AND `fetched is not None` AND
  `fetched["name"] == "Widget Pro"`

**What's tested beyond `argument_fidelity`:**
- `ENT-4728` is harder to fabricate than the original bare `42` — model
  must faithfully extract the alphanumeric ID from lookup's response
- Decoy entity (Widget Basic / `ENT-3301`) means wrong name → wrong ID →
  wrong details → validation fails. Lambda version had no decoy.
- Same difficulty gradient as lambda pair: `argument_fidelity` < `tool_selection`
  (no distractors vs 6 distractors). Stateful versions should widen the gap.

**Tags:** `["stateful", "model_quality"]` | **Ideal iterations:** 3

#### 6. `sequential_reasoning_stateful` — Medical Records DB

Mirrors `sequential_reasoning`. Same 4-tool, 4-step chain: identify patient
→ get records → check interactions → recommend. The stateful addition:
three-link data dependency chain where each tool's argument depends on the
previous tool's response. Wrong patient → wrong medication → wrong
interactions → validation fails.

**Backend class: `MedicalRecordsDB`**

```python
class MedicalRecordsDB:
    def __init__(self):
        self.patients = {
            "john doe": {
                "patient_id": "PT-7829", "dob": "1985-03-14",
                "blood_type": "O+",
            },
            "jane smith": {
                "patient_id": "PT-4215", "dob": "1990-07-22",
                "blood_type": "A-",
            },
        }
        self.records = {
            "PT-7829": {
                "last_visit": "2024-11-02", "diagnosis": "hypertension",
                "medication": "lisinopril 10mg",
            },
            "PT-4215": {
                "last_visit": "2024-10-18", "diagnosis": "type 2 diabetes",
                "medication": "metformin 500mg",
            },
        }
        self.interactions = {
            "lisinopril": (
                "Interactions: lisinopril + ibuprofen = risk of kidney "
                "damage. lisinopril + potassium supplements = "
                "hyperkalemia risk."
            ),
            "metformin": (
                "Interactions: metformin + alcohol = lactic acidosis risk. "
                "metformin + contrast dye = kidney injury risk."
            ),
        }
        self.identified = None
        self.records_fetched = None
        self.interactions_checked = None

    def identify_patient(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.patients:
            p = self.patients[key]
            self.identified = p
            return (
                f"Patient ID: {p['patient_id']}, DOB: {p['dob']}, "
                f"Blood type: {p['blood_type']}"
            )
        return f"No patient found for '{name}'."

    def get_records(self, patient_id: str) -> str:
        pid = patient_id.strip()
        if pid in self.records:
            r = self.records[pid]
            self.records_fetched = r
            return (
                f"Records: Last visit {r['last_visit']}, "
                f"Diagnosis: {r['diagnosis']}, "
                f"Medication: {r['medication']}"
            )
        return f"No records found for patient ID '{patient_id}'."

    def check_interactions(self, medication: str) -> str:
        # "lisinopril 10mg" → "lisinopril"
        key = medication.strip().lower().split()[0]
        if key in self.interactions:
            self.interactions_checked = key
            return self.interactions[key]
        return f"No interaction data found for '{medication}'."

    def recommend(self, patient_id: str, findings: str) -> str:
        return findings  # echo-back terminal
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `identify_patient` | Identify a patient by name. | `name` (string): Patient's full name |
| `get_records` | Get medical records for a patient. | `patient_id` (string): The patient's ID |
| `check_interactions` | Check drug interactions for a medication. | `medication` (string): Medication name |
| `recommend` | Provide a recommendation based on findings. | `patient_id` (string): The patient's ID; `findings` (string): The findings to base the recommendation on |

**System prompt:** "You are a medical assistant. Identify the patient,
retrieve their records, check drug interactions for their current
medication, then provide a recommendation."

**User message:** "Check drug interactions for patient John Doe's current
medication."

**Validation:**
- `validate` (terminal args): `findings` mentions "lisinopril", plus
  "kidney" or "ibuprofen", plus "hyperkalemia" or "potassium"
- `validate_state`: `identified is not None` AND
  `records_fetched is not None` AND `interactions_checked == "lisinopril"`

**What's tested beyond `sequential_reasoning`:**
- Three-link extraction chain: name → `PT-7829` → `lisinopril 10mg` →
  interaction query. Each step must extract from the previous response.
- `check_interactions` fuzzy-matches on first word — "lisinopril 10mg" and
  "lisinopril" both work, but "hypertension" (the diagnosis) fails
- Decoy patient (Jane Smith / PT-4215 / metformin) means wrong patient_id →
  wrong medication → completely different interactions → validation fails
- Lambda version returned the same canned response regardless of arguments;
  here, wrong arguments cascade through the entire chain

**Tags:** `["stateful", "model_quality"]` | **Ideal iterations:** 4

#### 7. `conditional_routing_stateful` — Incident Triage

Mirrors `conditional_routing`. Same 5-tool incident response workflow:
get alert → check metrics → check logs → check deployment → diagnose.
Required steps: `get_alert`, `check_metrics` (same as original). The
stateful addition: `service` param selects between two services with
completely different incident patterns. auth-service is a decoy — brief
self-resolving latency blip, deploy 5 days old, no breaking change.

**Backend class: `IncidentTriage`**

```python
class IncidentTriage:
    def __init__(self):
        self.alerts = {
            "payments-service": {
                "alert_id": "P1-8842", "type": "Error Rate Threshold",
                "error_rate": "12.4%", "threshold": "2%",
                "endpoint": "/api/v2/charge", "duration_min": 18,
                "triggered": "2025-01-15 14:23:07 UTC",
                "last_deploy": "2025-01-15 14:04:51 UTC",
            },
            "auth-service": {
                "alert_id": "P2-3317", "type": "Latency Threshold",
                "error_rate": "0.8%", "threshold": "500ms",
                "endpoint": "/api/v1/login", "duration_min": 5,
                "triggered": "2025-01-14 09:15:00 UTC",
                "last_deploy": "2025-01-10 11:00:00 UTC",
            },
        }
        self.metrics = {
            "payments-service": (
                "Metrics for payments-service (last 60 min):\n"
                "  14:00 — error_rate: 0.3%, latency_p99: 120ms, "
                "cpu: 45%, mem: 62%\n"
                "  14:05 — error_rate: 0.4%, latency_p99: 118ms, "
                "cpu: 44%, mem: 61%\n"
                "  14:10 — error_rate: 8.1%, latency_p99: 940ms, "
                "cpu: 47%, mem: 63%\n"
                "  14:15 — error_rate: 11.2%, latency_p99: 1850ms, "
                "cpu: 51%, mem: 64%\n"
                "  14:20 — error_rate: 12.4%, latency_p99: 2100ms, "
                "cpu: 52%, mem: 65%\n"
                "\n"
                "Note: Error spike begins between 14:05 and 14:10. No "
                "significant\nCPU or memory change. Latency correlates "
                "with error rate."
            ),
            "auth-service": (
                "Metrics for auth-service (last 60 min):\n"
                "  09:10 — error_rate: 0.1%, latency_p99: 80ms, "
                "cpu: 30%, mem: 45%\n"
                "  09:15 — error_rate: 0.8%, latency_p99: 620ms, "
                "cpu: 32%, mem: 46%\n"
                "  09:20 — error_rate: 0.2%, latency_p99: 95ms, "
                "cpu: 30%, mem: 45%\n"
                "\n"
                "Note: Brief latency spike at 09:15, self-resolved "
                "within 5 minutes."
            ),
        }
        self.logs = {
            "payments-service": (
                "Recent logs for payments-service (last 30 min):\n"
                "  14:08:12 WARN  [HttpClient] Retry attempt 1 for "
                "upstream call\n"
                "  14:08:15 WARN  [HttpClient] Retry attempt 2 for "
                "upstream call\n"
                "  14:09:01 ERROR [PaymentProcessor] Transaction failed: "
                "unexpected response format\n"
                "  14:09:03 ERROR [PaymentProcessor] Transaction failed: "
                "unexpected response format\n"
                "  14:11:44 WARN  [ConnectionPool] Pool utilization "
                "at 78%\n"
                "  14:14:22 ERROR [PaymentProcessor] Transaction failed: "
                "unexpected response format\n"
                "  14:18:33 WARN  [HttpClient] Retry attempt 1 for "
                "upstream call\n"
                "  (247 similar entries omitted)"
            ),
            "auth-service": (
                "Recent logs for auth-service (last 30 min):\n"
                "  09:14:55 WARN  [RateLimiter] Spike in login attempts "
                "from subnet 10.0.3.0/24\n"
                "  09:15:01 WARN  [RateLimiter] Throttling enabled\n"
                "  09:15:42 INFO  [RateLimiter] Throttling cleared, "
                "traffic normal\n"
                "  (3 entries total)"
            ),
        }
        self.deploys = {
            "payments-service": (
                "Last deployment to payments-service:\n"
                "  Deploy ID: deploy-a7f3e2\n"
                "  Timestamp: 2025-01-15 14:04:51 UTC\n"
                "  Author: jenkins-ci (triggered by merge PR #1147)\n"
                "  Changes: Updated payment gateway SDK from v3.8.1 "
                "to v4.0.0\n"
                "  Changelog note: \"v4.0.0 — Breaking change: response "
                "schema updated, 'transaction_id' field moved from root "
                "to 'data.transaction_id'\"\n"
                "  Rollback available: Yes (deploy-b82c1a, v3.8.1)"
            ),
            "auth-service": (
                "Last deployment to auth-service:\n"
                "  Deploy ID: deploy-c4d9f1\n"
                "  Timestamp: 2025-01-10 11:00:00 UTC\n"
                "  Author: jenkins-ci (triggered by merge PR #1098)\n"
                "  Changes: Updated logging library to v2.1.0\n"
                "  Rollback available: Yes (deploy-d1e2a3)"
            ),
        }
        self.alert_checked = None
        self.metrics_checked = None
        self.logs_checked = None
        self.deploy_checked = None

    def get_alert(self, service: str) -> str:
        key = service.strip().lower()
        if key in self.alerts:
            a = self.alerts[key]
            self.alert_checked = key
            return (
                f"Alert: {a['alert_id']} — {service}\n"
                f"Type: {a['type']}\n"
                f"Current error rate: {a['error_rate']} "
                f"(threshold: {a['threshold']})\n"
                f"Affected endpoint: {a['endpoint']}\n"
                f"Duration: {a['duration_min']} minutes\n"
                f"Triggered: {a['triggered']}\n"
                f"Last deploy: {a['last_deploy']}"
            )
        return f"No alert found for service '{service}'."

    def check_metrics(self, service: str) -> str:
        key = service.strip().lower()
        if key in self.metrics:
            self.metrics_checked = key
            return self.metrics[key]
        return f"No metrics found for service '{service}'."

    def check_logs(self, service: str) -> str:
        key = service.strip().lower()
        if key in self.logs:
            self.logs_checked = key
            return self.logs[key]
        return f"No logs found for service '{service}'."

    def check_deployment(self, service: str) -> str:
        key = service.strip().lower()
        if key in self.deploys:
            self.deploy_checked = key
            return self.deploys[key]
        return f"No deployment info for service '{service}'."

    def diagnose(self, diagnosis: str, action: str) -> str:
        return f"Diagnosis: {diagnosis} | Action: {action}"
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `get_alert` | Get details for the current P1 alert. | `service` (string): Service name |
| `check_metrics` | Get time-series system metrics for a service. | `service` (string): Service name |
| `check_logs` | Get recent log entries for a service. | `service` (string): Service name |
| `check_deployment` | Get details of the last deployment to a service. | `service` (string): Service name |
| `diagnose` | Submit a root cause diagnosis and recommended action. | `diagnosis` (string): The root cause diagnosis; `action` (string): The recommended action |

**System prompt:** "You are an infrastructure incident responder. Use the
available tools to investigate the P1 alert, determine the root cause,
and recommend an action. Call diagnose when you have enough evidence to
identify the root cause."

**User message:** "We got a P1 alert on the payments service. Diagnose
the root cause and recommend an action."

**Validation:**
- `validate` (terminal args): `diagnosis` + `action` mentions "sdk" or
  "v4.0.0" or "gateway", plus "rollback" or "revert", plus "response" or
  "schema" or "transaction_id"
- `validate_state`: `alert_checked == "payments-service"` AND
  `metrics_checked == "payments-service"` AND
  `deploy_checked == "payments-service"`

**What's tested beyond `conditional_routing`:**
- `service` param now selects between two services with completely
  different incident patterns. auth-service is a decoy: brief
  self-resolving latency blip, deploy 5 days old, no breaking change —
  nothing points to a rollback.
- Same temporal correlation challenge as original (deploy at 14:04:51,
  error spike between 14:05–14:10), but data comes from keyed lookups
- State validates the model investigated the *right* service consistently
  across all tools — not just that the final diagnosis is correct
- `check_logs` not required but provides supporting evidence
  ("unexpected response format" corroborates the SDK schema change)

**Tags:** `["stateful", "model_quality", "reasoning"]` | **Ideal iterations:** 4

#### 8. `data_gap_recovery_stateful` — HR Records System

Mirrors `data_gap_recovery`. Same 7-tool HR workspace with breadcrumb
hints and dead-end redirects. Required step: `get_employee` only. The
stateful addition: `employee_id` extracted from `get_employee` must be
threaded into all ID-based tools. Decoy employee (James Liu / E-2234)
returns completely different data per tool.

**Backend class: `HRRecordsSystem`**

```python
class HRRecordsSystem:
    def __init__(self):
        self.employees = {
            "sarah chen": {
                "employee_id": "E-1847", "department": "Engineering",
                "title": "Senior Backend Engineer",
                "start_date": "2019-03-15", "manager": "David Park",
            },
            "james liu": {
                "employee_id": "E-2234", "department": "Marketing",
                "title": "Content Strategist",
                "start_date": "2021-08-01", "manager": "Maria Santos",
            },
        }
        self.security = {
            "E-1847": {
                "clearance": "L3 — Confidential",
                "granted": "2021-06-10", "sponsor": "David Park",
                "expires": "2025-12-01",
                "access_groups": "payments-prod, internal-apis, staging-*",
            },
            "E-2234": {
                "clearance": "L1 — Public",
                "granted": "2021-08-01", "sponsor": "Maria Santos",
                "expires": "2026-08-01",
                "access_groups": "marketing-tools, cms-prod",
            },
        }
        self.onboarding = {
            "E-1847": {
                "emergency_contact": (
                    "Michael Chen (spouse) — (555) 867-5309"
                ),
                "onboarding_date": "2019-03-15",
            },
            "E-2234": {
                "emergency_contact": (
                    "Linda Liu (mother) — (555) 234-5678"
                ),
                "onboarding_date": "2021-08-01",
            },
        }
        self.employee_looked_up = None
        self.security_fetched = None
        self.onboarding_fetched = None

    def get_employee(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.employees:
            e = self.employees[key]
            self.employee_looked_up = e
            return (
                f"Employee Record — {name.title()} "
                f"(ID: {e['employee_id']})\n"
                f"Department: {e['department']}\n"
                f"Title: {e['title']}\n"
                f"Start Date: {e['start_date']}\n"
                f"Manager: {e['manager']}\n"
                f"Clearance Level: [RESTRICTED — request via "
                f"security_audit]\n"
                f"Emergency Contact: [MIGRATED — see historical "
                f"employee records]"
            )
        return f"No employee found for '{name}'."

    def security_audit(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid in self.security:
            s = self.security[eid]
            self.security_fetched = eid
            return (
                f"Security Audit Record — {eid}\n"
                f"Clearance Level: {s['clearance']}\n"
                f"Granted: {s['granted']}\n"
                f"Sponsor: {s['sponsor']}\n"
                f"Expires: {s['expires']}\n"
                f"Access Groups: {s['access_groups']}"
            )
        return f"No security audit record for '{employee_id}'."

    def personnel_file(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid not in {"E-1847", "E-2234"}:
            return f"No personnel file for '{employee_id}'."
        return (
            f"Personnel File — {eid}\n"
            f"Contains: compensation history, performance reviews, "
            f"PTO balance.\n"
            f"Note: Emergency contact information is not stored in "
            f"personnel files.\n"
            f"It was moved to HR onboarding records during the 2020 "
            f"system migration."
        )

    def onboarding_records(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid in self.onboarding:
            o = self.onboarding[eid]
            self.onboarding_fetched = eid
            return (
                f"HR Onboarding Record — {eid}\n"
                f"Original Onboarding Date: {o['onboarding_date']}\n"
                f"Emergency Contact: {o['emergency_contact']}"
            )
        return f"No onboarding record for '{employee_id}'."

    def compliance_check(self, employee_id: str) -> str:
        return (
            f"Compliance Status — {employee_id}: All mandatory "
            f"trainings complete. No outstanding compliance items."
        )

    def hr_directory(self, name: str) -> str:
        key = name.strip().lower()
        if key in self.employees:
            e = self.employees[key]
            return (
                f"HR Directory Entry — {name.title()}\n"
                f"Department: {e['department']} | "
                f"Reports to: {e['manager']}\n"
                f"Status: Active | Full-time"
            )
        return f"No directory entry for '{name}'."

    def submit_report(self, employee_name: str, report: str) -> str:
        return f"Report submitted for {employee_name}."
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `get_employee` | Look up an employee record by name. | `name` (string): Employee's full name |
| `security_audit` | Query security audit records for an employee. | `employee_id` (string): Employee's ID |
| `personnel_file` | Access an employee's personnel file. | `employee_id` (string): Employee's ID |
| `onboarding_records` | Access HR onboarding records for an employee. | `employee_id` (string): Employee's ID |
| `compliance_check` | Check compliance status for an employee. | `employee_id` (string): Employee's ID |
| `hr_directory` | Look up an employee in the HR directory. | `name` (string): Employee's name |
| `submit_report` | Submit a completed employee profile report. | `employee_name` (string): Employee's full name; `report` (string): The complete profile report |

**System prompt:** "You are an HR systems assistant. Use the available
tools to gather the requested employee information and submit a complete
profile report."

**User message:** "Pull together a complete profile for Sarah Chen — we
need her clearance level and emergency contact for the onboarding audit."

**Validation:**
- `validate` (terminal args): report mentions "engineering", plus "l3"
  or "confidential", plus "michael" and ("867-5309" or "spouse")
- `validate_state`: `employee_looked_up is not None` AND
  `security_fetched == "E-1847"` AND `onboarding_fetched == "E-1847"`

**What's tested beyond `data_gap_recovery`:**
- `employee_id` (`E-1847`) must be extracted from `get_employee` and
  threaded into `security_audit`, `personnel_file`, `onboarding_records`
- Decoy employee (James Liu / E-2234) returns L1 clearance and different
  emergency contact — wrong ID cascades silently through all tools
- Same breadcrumb/dead-end challenge: clearance hint →
  `security_audit`, emergency contact → `personnel_file` (dead-end,
  redirects to `onboarding_records`). But now wrong ID at any step →
  wrong person's data → validation fails.
- State validates the model queried the *right* employee's security and
  onboarding records, not just that the report content is correct

**Tags:** `["stateful", "model_quality", "reasoning"]` | **Ideal iterations:** 5

#### 9. `relevance_detection_stateful` — Travel Booking System

Mirrors `relevance_detection`. Same 5-tool travel namespace (4 travel
tools + `decline`). Same off-topic user message. Same expected behavior:
model should call `decline` without touching any travel tools. The
stateful addition: the 4 non-terminal tools actually record state
(bookings made, forecasts fetched, currencies converted). Validation
checks that **no state was mutated** — the inverse of every other
stateful scenario.

**Backend class: `TravelBookingSystem`**

```python
class TravelBookingSystem:
    def __init__(self):
        self.forecasts_fetched = []
        self.flights_booked = []
        self.hotels_checked = []
        self.conversions = []

    def get_forecast(self, city: str) -> str:
        self.forecasts_fetched.append(city.strip().lower())
        return f"Forecast for {city}: 22°C, partly cloudy."

    def book_flight(self, origin: str, destination: str) -> str:
        self.flights_booked.append(
            (origin.strip().lower(), destination.strip().lower())
        )
        return f"Flight booked: {origin} → {destination}."

    def check_hotel(self, city: str, checkin: str) -> str:
        self.hotels_checked.append(city.strip().lower())
        return f"3 hotels available in {city} for {checkin}."

    def convert_currency(
        self, amount: str, from_currency: str, to_currency: str,
    ) -> str:
        self.conversions.append(
            (amount, from_currency.upper(), to_currency.upper())
        )
        return f"{amount} {from_currency} = 0.00 {to_currency}."

    def decline(self, reason: str) -> str:
        return reason
```

**Tool schemas (what the model sees):**

| Tool | Description | Params |
|------|-------------|--------|
| `get_forecast` | Get weather forecast for a city. | `city` (string): The city name |
| `book_flight` | Book a flight between two cities. | `origin` (string): Departure city; `destination` (string): Arrival city |
| `check_hotel` | Check hotel availability in a city. | `city` (string): The city to check; `checkin` (string): Check-in date |
| `convert_currency` | Convert an amount between currencies. | `amount` (string): The amount to convert; `from_currency` (string): Source currency code; `to_currency` (string): Target currency code |
| `decline` | Call this when none of the available tools are relevant to the user's request. | `reason` (string): Brief explanation of why no tool is appropriate |

**System prompt:** "You are a helpful assistant. You have access to
travel-related tools. If the user's request cannot be answered using
the available tools, call the decline tool to explain why. Do NOT call
a tool unless it is directly relevant."

**User message:** "What is the square root of 144?"

**Validation:**
- `validate` (terminal args): `reason` is non-empty
- `validate_state`: `forecasts_fetched == []` AND
  `flights_booked == []` AND `hotels_checked == []` AND
  `conversions == []`

**What's tested beyond `relevance_detection`:**
- Lambda version can't detect if the model called `book_flight` before
  eventually calling `decline` — the canned lambda doesn't record it.
  Stateful version catches any stray tool call via state mutation.
- This is the one scenario where state should remain **untouched**.
  Every other stateful scenario validates that state *was* mutated
  correctly. This one validates the opposite.
- Same difficulty as original — easiest scenario. The stateful layer
  is a stricter safety net, not a harder challenge.

**Tags:** `["stateful", "model_quality"]` | **Ideal iterations:** 1

## Future: `ToolSoftError` — graceful argument errors

Currently forge has two error modes: exceptions (runner retries) and normal
returns (runner moves on). A tool returning `"No user found for 'xyz'"` is a
200 OK — the runner marks the step complete and the model proceeds with bad
data. There's no middle ground.

Real platforms have HTTP status codes (404 vs 500 vs 200). The LLM analog:
a `ToolSoftError` exception that means "your call was well-formed but the
data didn't resolve." The runner catches it, sends the message back as a
tool result, and does NOT mark the step as completed — letting the model
retry with different args within the existing retry budget.

```python
from forge.core.workflow import ToolSoftError

def get_user(self, user_id: str) -> str:
    if user_id not in self.db:
        raise ToolSoftError(f"No user found for '{user_id}'.")
    return self.db[user_id]
```

### Why this matters — the `_resolve_service` lesson

The stateful scenarios exposed this gap immediately. The user message says
"payments service" but the backend keys are `"payments-service"` (hyphenated).
The model extracts `"payments"`, passes it to every tool, and every lookup
misses — returning `"No alert found for service 'payments'"` as a normal
200-style return. The runner marks `get_alert` as completed and the model
barrels ahead with no data.

We fixed this by adding `_resolve_service()` — a fuzzy matcher that tries
appending `-service`, stripping `"the "`, etc. But that's the scenario
author doing manually what `ToolSoftError` would let the runner do
generically. With `ToolSoftError`, the backend stays strict: `raise
ToolSoftError("No alert found for 'payments'")`. The runner catches it,
sends the message back as the tool result, keeps `get_alert` uncompleted,
and the model retries with `"payments-service"`.

In a real-world workflow where the dev writes strict lookups — which is the
natural thing to do — every key-miss becomes a silent bad-data cascade.
`ToolSoftError` turns those into recoverable retries without requiring
every backend to implement its own fuzzy matching.

### Eval tiers

The stateful scenarios deliberately test the harder path: model must get it
right the first time (with fuzzy backends softening arg formatting only).
`ToolSoftError` would add a sixth eval tier:

1. **Lambda** — forge plumbing baseline
2. **Ablated lambda** — how much do guardrails help
3. **Stateful** (fuzzy backends) — model reasoning quality
4. **Ablated stateful** — guardrails × reasoning
5. **Stateful + ToolSoftError** (strict backends) — how much does
   retry-on-miss recover

Tier 5 is the interesting comparison: if `ToolSoftError` closes most of the
gap between fuzzy and strict backends, it validates the feature as a
real-world safety net rather than just a nice abstraction.

### Implementation status

Post-launch feature. Tiers 1–4 are runnable today without any code changes.
Tier 5 requires `ToolSoftError` in the runner + strict backend variants of
the stateful scenarios.

Blog post angle: HTTP error taxonomy was designed for deterministic clients.
LLMs need a different error taxonomy where "valid request, no result" is
a recoverable state, not a silent pass-through.

## Future scenario idea: `information_loss` — silent write corruption

Not part of this batch. Captures a different failure mode: the backend
confirms success even when the model wrote garbage.

**Concept:** Simple DB with schema, read, write, compute, and submit tools.
The user asks to update a record and compute something from the new value.

- `get_schema()` → returns table structure (columns, types, constraints)
- `read_record(table, id)` → returns current row
- `write_record(table, id, data)` → always returns "Row updated
  successfully" — even if data has wrong types, missing fields, or bad
  column names (silently coerces or drops)
- `compute(query)` → runs calculation against *actual stored state*
- `submit(result)` → terminal

**Example user message:** "Update employee E-1001's salary to 75000 and
compute their new monthly take-home after 22% tax."

**What makes it hard:**
- `write_record` never complains — wrong column name silently ignored,
  string where int expected silently stored as-is
- `compute` operates on actual stored state, not what the model *thinks*
  it wrote — if the write was malformed, compute returns wrong results
- A smart model would `read_record` after writing to verify state matches
  intent (read-after-write self-verification)

**What it tests that nothing else does:**
- Self-verification behavior (read-after-write)
- Schema compliance without explicit validation errors
- The gap between "tool returned success" and "tool did what I wanted"

**Validation:** Strict — check that final stored state is correct AND
compute result is correct. Models that don't self-verify can still pass if
they got the write right on the first try. Could optionally track whether
`read_record` was called after `write_record` as a behavioral observation
(not a pass/fail gate).

**Relationship to `ToolSoftError`:** Orthogonal. `ToolSoftError` is about
the backend signaling "your args didn't resolve." This scenario is about
the backend *not* signaling anything — the model has to figure it out
itself. Both test different failure modes of the same underlying problem:
LLMs can't observe the side effects of their tool calls.

## What doesn't change

- **Compaction scenarios** (`compaction_stress`, `phase2_compaction`) — not
  mirrored. They test compaction mechanics, not tool correctness. Stateful
  tools would add noise without testing what matters.
- **`WorkflowRunner`** — no changes. It already calls `fn(**tc.args)` and
  handles typed args.
- **`ToolParam.type`** — already supports `"integer"`, `"boolean"`, etc.
  Stateful scenarios use proper types in their `ToolSpec` definitions.
- **Reporting pipeline** (`metrics.py`, `report.py`, `batch_eval.py`) —
  scenarios are independent. New scenarios appear as new columns automatically.
- **Existing ablation data** — remains valid. Old lambda scenarios still run.
  New stateful scenarios get their own ablation runs independently.

## Sequencing

1. ~~Add `validate_state` to `EvalScenario`, wire in `eval_runner`~~ — **Done** (ebb65d6)
2. ~~Add `build_workflow` factory to `EvalScenario`, wire in `eval_runner`~~ — **Done** (ebb65d6)
3. ~~Convert existing 11 scenarios from `make_tool` to direct `ToolDef`~~ — **Done** (6a7de6d)
3b. ~~Split `scenarios.py` into `scenarios/` package~~ — **Done** (01b3209)
    - `_base.py`: `EvalScenario` dataclass + `_check()` + `_placeholder_workflow()` helpers
    - `_plumbing.py`: `basic_2step`, `sequential_3step`, `compaction_stress`, `error_recovery`
    - `_model_quality.py`: `tool_selection`, `argument_fidelity`, `sequential_reasoning`, `conditional_routing`, `data_gap_recovery`
    - `_compaction.py`: `phase2_compaction`, `relevance_detection`
    - `__init__.py`: re-exports everything, all import sites unchanged
4. ~~Build 9 stateful backend classes + scenarios~~ — **Done** (69959f7)
    - `_stateful_plumbing.py`: `CountryFactsDB`, `SalesPipeline`, `InventorySystem`
    - `_stateful_model_quality.py`: `UserPermissionsDB`, `EntityRegistry`, `MedicalRecordsDB`, `IncidentTriage`, `HRRecordsSystem`
    - `_stateful_compaction.py`: `TravelBookingSystem`
    - Fuzzy service name resolution in `IncidentTriage._resolve_service()`
    - All backends use `.lower()` normalization on user-derived string params
5. Unit tests for backend classes (state transitions, edge cases) — deferred, smoke-tested manually
6. ~~Smoke test with one model~~ — **Done.** 9/11 correct on Ministral 8B; failures are model reasoning (conditional_routing skipping `check_deployment`), not scenario bugs.

All steps complete. Full ablation run pending.

## References

- BFCL stateful backends: `ref_docs/BFCL_REFERENCE.md` (Simulated Backends section)
- BFCL integration concept: `docs/decisions/BFCL_INTEGRATION_CONCEPT.md`
- Current scenarios: `tests/eval/scenarios/` (package)
- Eval runner: `tests/eval/eval_runner.py`
- Workflow/ToolDef: `src/forge/core/workflow.py`
