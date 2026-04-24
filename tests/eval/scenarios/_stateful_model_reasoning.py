"""Stateful advanced model reasoning scenarios — designed to weed out
top-tier models. Mirrors _model_reasoning.py with class-backed state
tracking for `validate_state` checks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _placeholder_workflow
from ._stateful_model_quality import (
    EmployeeIdParams,
    EmployeeNameParams,
    HRRecordsSystem,
    SubmitReportParams,
)


# ── Pydantic param models (argument_transformation) ─────────────


class NoParams(BaseModel):
    pass

class QuarterYearParams(BaseModel):
    quarter: str = Field(description="Fiscal quarter, e.g. 'Q4'")
    year: int = Field(description="Fiscal year, e.g. 2024")

class VendorNameParams(BaseModel):
    vendor_name: str = Field(description="Vendor name (case-sensitive in our records)")

class CurrencyConvertParams(BaseModel):
    amount: float = Field(description="Amount to convert")
    from_currency: str = Field(description="Source currency code, e.g. 'EUR'")
    to_currency: str = Field(description="Target currency code, e.g. 'USD'")

class CategorizeExpenseParams(BaseModel):
    amount: float = Field(description="Expense amount")
    category: str = Field(description="Expense category")

class TransactionIdParams(BaseModel):
    transaction_id: str = Field(description="Transaction ID, e.g. 'TX-1001'")

class SubmitAuditReportParams(BaseModel):
    transaction_ids: str = Field(description="Comma-separated transaction IDs to flag")
    total_flagged_usd: str = Field(description="Total flagged amount in USD")
    top_vendor: str = Field(description="Vendor of the single largest flagged transaction")


# ── Backend: HRRecordsSystemExtended (subclass of HRRecordsSystem) ─


class HRRecordsSystemExtended(HRRecordsSystem):
    """Extends HRRecordsSystem with compensation_v2, project_assignments, and
    deprecated/legacy endpoints used by data_gap_recovery_extended."""

    def __init__(self) -> None:
        super().__init__()
        self.compensation = {
            "E-1847": {"name": "Sarah Chen", "band": "B7",
                       "effective": "2024-04-01", "next_review": "2025-04-01"},
            "E-2234": {"name": "James Liu", "band": "B4",
                       "effective": "2024-08-01", "next_review": "2025-08-01"},
        }
        self.project_assignments_data = {
            "E-1847": {
                "name": "Sarah Chen",
                "active_groups": "payments-prod-v3, internal-apis-v2, staging-test",
                "primary": "Payments Platform Modernization",
                "secondary": "Internal Tooling Sprint",
            },
            "E-2234": {
                "name": "James Liu",
                "active_groups": "marketing-tools-v2, cms-prod-v3",
                "primary": "Brand Refresh",
                "secondary": "Content Migration",
            },
        }
        self.compensation_v2_fetched: str | None = None
        self.project_assignments_fetched: str | None = None

    def get_employee(self, name: str) -> str:
        # Override to redact salary/access fields. No tool-name hints —
        # model must discover the right downstream tools by name and
        # infer routing from domain language in personnel_file.
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
                f"Office: {e['office']}\n"
                f"Manager: {e['manager']}\n"
                f"Clearance Level: [RESTRICTED]\n"
                f"Emergency Contact: [unavailable in this record]\n"
                f"Salary Band: [PROTECTED]\n"
                f"Access Groups: [managed externally]"
            )
        return f"No employee found for '{name}'."

    def security_audit(self, employee_id: str) -> str:
        # Override to drop the project_assignments redirect — the
        # provisioned access groups now look like a complete answer
        # (soft trap; missing -v3/-v2 suffixes the validator wants).
        eid = employee_id.strip()
        if eid in self.security:
            s = self.security[eid]
            self.security_fetched = eid
            return (
                f"Security Audit Record — {eid} ({s['name']})\n"
                f"Clearance Level: {s['clearance']}\n"
                f"Granted: {s['granted']}\n"
                f"Sponsor: {s['sponsor']}\n"
                f"Last Review: {s['last_review']}\n"
                f"Expires: {s['expires']}\n"
                f"Access Groups: {s['access_groups']}"
            )
        return f"No security audit record for '{employee_id}'."

    def personnel_file(self, employee_id: str) -> str:
        # Override to use domain language only (no explicit tool names).
        # Model must infer compensation_v2 vs compensation_history from
        # "current pay system" and find onboarding_records by name.
        eid = employee_id.strip()
        if eid not in {"E-1847", "E-2234"}:
            return f"No personnel file for '{employee_id}'."
        return (
            f"Personnel File — {eid}\n"
            f"Contains: performance reviews, PTO balance.\n"
            f"Note: Compensation data is in our current pay system "
            f"(migrated 2024).\n"
            f"Note: Emergency contact records are with HR onboarding "
            f"(2020 migration)."
        )

    def compensation_v2(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid in self.compensation:
            c = self.compensation[eid]
            self.compensation_v2_fetched = eid
            return (
                f"Compensation v2 — {eid} ({c['name']})\n"
                f"Salary Band: {c['band']}\n"
                f"Effective: {c['effective']}\n"
                f"Next Review: {c['next_review']}"
            )
        return f"No compensation_v2 record for '{employee_id}'."

    def project_assignments(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid in self.project_assignments_data:
            p = self.project_assignments_data[eid]
            self.project_assignments_fetched = eid
            return (
                f"Project Assignments — {eid} ({p['name']})\n"
                f"Active Access Groups: {p['active_groups']}\n"
                f"Primary Project: {p['primary']}\n"
                f"Secondary: {p['secondary']}"
            )
        return f"No project assignments for '{employee_id}'."

    def compensation_history(self, employee_id: str) -> str:
        return (
            "compensation_history — DEPRECATED endpoint.\n"
            "This service was retired in 2024. Use compensation_v2 for "
            "current salary band data."
        )

    def access_control(self, employee_id: str) -> str:
        return (
            "access_control — DEPRECATED endpoint.\n"
            "Replaced by project_assignments for current active access "
            "group data. Static provisioning records remain in security_audit."
        )

    def legacy_records(self, employee_id: str) -> str:
        eid = employee_id.strip()
        if eid not in {"E-1847", "E-2234"}:
            return f"No legacy archive for '{employee_id}'."
        # Plausible-but-stale data with embedded redirect note.
        return (
            f"Legacy Archive — {eid}\n"
            f"Source: legacy_v1 (READ-ONLY — not synced with current "
            f"systems)\n"
            f"Salary Band: B5\n"
            f"Access Groups: legacy-readonly\n"
            f"Note: legacy_v1 stopped syncing after the 2024 systems "
            f"migration. For current salary band, query "
            f"compensation_v2. For current access groups, query "
            f"project_assignments. Data above is provided for "
            f"historical audit only."
        )


def _validate_data_gap_recovery_extended_stateful(
    args: dict[str, Any],
) -> bool:
    text = f"{args.get('employee_name', '')} {args.get('report', '')}".lower()
    has_dept      = "engineering" in text
    has_clearance = "l3" in text or "confidential" in text
    has_contact   = "michael" in text and ("867-5309" in text or "spouse" in text)
    has_salary    = "b7" in text
    has_groups    = "payments-prod-v3" in text or "internal-apis-v2" in text
    return has_dept and has_clearance and has_contact and has_salary and has_groups


def _build_data_gap_recovery_extended_stateful() -> tuple[Workflow, callable]:
    db = HRRecordsSystemExtended()
    tools: dict[str, ToolDef] = {
        "get_employee": ToolDef(
            spec=ToolSpec(
                name="get_employee",
                description="Look up an employee record by name.",
                parameters=EmployeeNameParams,
            ),
            callable=lambda **kw: db.get_employee(kw["name"]),
        ),
        "security_audit": ToolDef(
            spec=ToolSpec(
                name="security_audit",
                description="Query security audit records for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.security_audit(kw["employee_id"]),
        ),
        "onboarding_records": ToolDef(
            spec=ToolSpec(
                name="onboarding_records",
                description="Access HR onboarding records for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.onboarding_records(kw["employee_id"]),
        ),
        "personnel_file": ToolDef(
            spec=ToolSpec(
                name="personnel_file",
                description="Access an employee's personnel file.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.personnel_file(kw["employee_id"]),
        ),
        "compensation_v2": ToolDef(
            spec=ToolSpec(
                name="compensation_v2",
                description="Current compensation system — salary band lookup.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.compensation_v2(kw["employee_id"]),
        ),
        "project_assignments": ToolDef(
            spec=ToolSpec(
                name="project_assignments",
                description="Current project assignments and active access groups for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.project_assignments(kw["employee_id"]),
        ),
        "compensation_history": ToolDef(
            spec=ToolSpec(
                name="compensation_history",
                description="Look up compensation history for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.compensation_history(kw["employee_id"]),
        ),
        "access_control": ToolDef(
            spec=ToolSpec(
                name="access_control",
                description="Look up access control entries for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.access_control(kw["employee_id"]),
        ),
        "legacy_records": ToolDef(
            spec=ToolSpec(
                name="legacy_records",
                description="Search legacy records archive for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.legacy_records(kw["employee_id"]),
        ),
        "compliance_check": ToolDef(
            spec=ToolSpec(
                name="compliance_check",
                description="Check compliance status for an employee.",
                parameters=EmployeeIdParams,
            ),
            callable=lambda **kw: db.compliance_check(kw["employee_id"]),
        ),
        "hr_directory": ToolDef(
            spec=ToolSpec(
                name="hr_directory",
                description="Look up an employee in the HR directory.",
                parameters=EmployeeNameParams,
            ),
            callable=lambda **kw: db.hr_directory(kw["name"]),
        ),
        "submit_report": ToolDef(
            spec=ToolSpec(
                name="submit_report",
                description="Submit a completed employee profile report.",
                parameters=SubmitReportParams,
            ),
            callable=lambda **kw: db.submit_report(kw.get("employee_name", ""), kw.get("report", "")),
        ),
    }
    workflow = Workflow(
        name="data_gap_recovery_extended_stateful",
        description=(
            "Build a complete employee profile with multi-hop data resolution "
            "across legacy and current systems."
        ),
        tools=tools,
        required_steps=["get_employee"],
        terminal_tool="submit_report",
        system_prompt_template=(
            "You are an HR systems assistant. Use the available tools to "
            "gather the requested employee information and submit a complete "
            "profile report. Read tool results carefully — some endpoints "
            "redirect to newer systems."
        ),
    )
    validate_state = lambda: (
        db.employee_looked_up is not None
        and db.security_fetched == "E-1847"
        and db.onboarding_fetched == "E-1847"
        and db.compensation_v2_fetched == "E-1847"
        and db.project_assignments_fetched == "E-1847"
    )
    return workflow, validate_state


data_gap_recovery_extended_stateful = EvalScenario(
    name="data_gap_recovery_extended_stateful",
    description="Stateful extended hint-chain — 5 facts, 2-3 hop chains, no direct hints, name-trap + status-marker + soft-trap lures.",
    workflow=_placeholder_workflow(
        "data_gap_recovery_extended_stateful", "submit_report", ["get_employee"],
    ),
    user_message=(
        "Pull together a complete profile for Sarah Chen — we need her "
        "clearance level, current emergency contact, latest salary band, "
        "and active access groups for the onboarding audit and access review."
    ),
    validate=_validate_data_gap_recovery_extended_stateful,
    build_workflow=_build_data_gap_recovery_extended_stateful,
    tags=["stateful", "advanced_reasoning", "reasoning", "model_quality"],
    ideal_iterations=8,
    max_iterations=20,
)


# ── Backend: ExpenseAuditSystem ─────────────────────────────────


_FX_EUR_TO_USD = 1.10

_APPROVED_VENDORS_STATEFUL = [
    "Acme Corp",
    "Globex Industries",
    "Initech Systems",
    "Umbrella Logistics",
    "Wayne Enterprises",
    "Stark Industries",
]

_Q4_2024_TRANSACTIONS_STATEFUL = [
    ("TX-1001", "2024-10-05", "Cyberdyne LLC",       7500,  "USD"),
    ("TX-1002", "2024-10-12", "Acme Corp",          12000,  "USD"),
    ("TX-1003", "2024-10-22", "Initech Systems",     8200,  "USD"),
    ("TX-1004", "2024-11-03", "Vandelay Imports",    3500,  "USD"),
    ("TX-1005", "2024-11-08", "Soylent Corp",        5000,  "USD"),
    ("TX-1006", "2024-11-14", "Pied Piper",          4800,  "EUR"),
    ("TX-1007", "2024-11-22", "Umbrella Logistics",  9400,  "USD"),
    ("TX-1008", "2024-12-02", "Wonka Industries",   11200,  "USD"),
    ("TX-1009", "2024-12-08", "ACME Corp",           6500,  "USD"),
    ("TX-1010", "2024-12-15", "Globex Industries",   5500,  "USD"),
    ("TX-1011", "2024-12-19", "Pied Piper",          2400,  "EUR"),
    ("TX-1012", "2024-12-22", "Stark Industries",   14800,  "USD"),
    ("TX-1013", "2024-12-28", "Wayne Enterprises",   7300,  "USD"),
]


class ExpenseAuditSystem:
    """Stateful Q4 expense audit backend. Tracks which tools were called
    so validate_state can verify the model used the right reasoning path
    (currency_convert for the EUR transaction, get_vendor_details for the
    case-mismatch alias) — not just guessed at the answer."""

    def __init__(self) -> None:
        self.transactions = list(_Q4_2024_TRANSACTIONS_STATEFUL)
        self.approved_vendors = list(_APPROVED_VENDORS_STATEFUL)
        # State tracking
        self.list_called_for: tuple[str, int] | None = None
        self.approved_called: bool = False
        self.vendor_details_called_for: set[str] = set()
        self.eur_conversion_called: bool = False
        self.submitted_args: dict[str, str] | None = None

    def list_transactions(self, quarter: str, year: int) -> str:
        q = str(quarter).strip().upper()
        y = int(year)
        if q == "Q4" and y == 2024:
            self.list_called_for = (q, y)
            lines = ["Q4 2024 Expense Transactions:"]
            for tid, date, vendor, amount, cur in self.transactions:
                lines.append(
                    f"  {tid} | {date} | vendor: {vendor:25s} | "
                    f"amount: {amount:>8,.2f} {cur}"
                )
            return "\n".join(lines)
        return f"No transactions found for {quarter} {year}."

    def get_approved_vendors(self) -> str:
        self.approved_called = True
        lines = ["Approved Vendors (canonical names — case sensitive):"]
        for v in self.approved_vendors:
            lines.append(f"  - {v}")
        return "\n".join(lines)

    def get_vendor_details(self, vendor_name: str) -> str:
        name = str(vendor_name).strip()
        self.vendor_details_called_for.add(name)
        if name == "ACME Corp":
            return (
                "Vendor Details — ACME Corp\n"
                "Status: registered trade-name alias of Acme Corp "
                "(unified entity 2023).\n"
                "All purchasing under this name applies to the Acme Corp "
                "master account and is governed by the same approval terms."
            )
        if name == "Acme Corp":
            return (
                "Vendor Details — Acme Corp\n"
                "Status: master account (legal entity since 1998).\n"
                "Trade-name aliases on file: ACME Corp."
            )
        if name in self.approved_vendors:
            return f"Vendor Details — {name}\nStatus: standard supplier (active)."
        return f"Vendor Details — {name}\nStatus: not found in vendor master."

    def currency_convert(
        self, amount: float, from_currency: str, to_currency: str,
    ) -> str:
        amt = float(amount)
        src = str(from_currency).strip().upper()
        dst = str(to_currency).strip().upper()
        if src == "EUR" and dst == "USD":
            self.eur_conversion_called = True
            converted = amt * _FX_EUR_TO_USD
            return (
                f"Conversion: {amt:,.2f} EUR = {converted:,.2f} USD "
                f"(rate: 1 EUR = {_FX_EUR_TO_USD} USD)"
            )
        if src == "USD" and dst == "EUR":
            converted = amt / _FX_EUR_TO_USD
            return (
                f"Conversion: {amt:,.2f} USD = {converted:,.2f} EUR "
                f"(rate: 1 EUR = {_FX_EUR_TO_USD} USD)"
            )
        if src == dst:
            return f"Conversion: {amt:,.2f} {src} = {amt:,.2f} {dst} (same currency)."
        return f"Unsupported conversion pair: {src} -> {dst}."

    def categorize_expense(self, amount: float, category: str) -> str:
        cat = str(category).strip()
        return (
            f"Categorization: {float(amount):,.2f} -> {cat} "
            f"(GL bucket: GL-{abs(hash(cat)) % 9000 + 1000})."
        )

    def lookup_transaction(self, transaction_id: str) -> str:
        tid = str(transaction_id).strip()
        for row in self.transactions:
            if row[0] == tid:
                _id, date, vendor, amount, cur = row
                return (
                    f"Transaction Record — {tid}\n"
                    f"Date: {date}\nVendor: {vendor}\n"
                    f"Amount: {amount:,.2f} {cur}"
                )
        return f"No transaction found for '{tid}'."

    def submit_audit_report(
        self, transaction_ids: str, total_flagged_usd: str, top_vendor: str,
    ) -> str:
        self.submitted_args = {
            "transaction_ids": str(transaction_ids),
            "total_flagged_usd": str(total_flagged_usd),
            "top_vendor": str(top_vendor),
        }
        return (
            f"Audit report submitted. "
            f"Flagged: {transaction_ids}; "
            f"total: {total_flagged_usd}; "
            f"top: {top_vendor}."
        )


_ARGTRANS_REQUIRED_IDS_STATEFUL = ("TX-1001", "TX-1005", "TX-1006", "TX-1008")
_ARGTRANS_TOTAL_TOKENS_STATEFUL = ("28,980", "28980")
_ARGTRANS_TOP_VENDOR_STATEFUL = "wonka"


def _validate_argument_transformation_stateful(args: dict[str, Any]) -> bool:
    ids_text = str(args.get("transaction_ids", ""))
    total_text = str(args.get("total_flagged_usd", "")).replace("$", "")
    vendor_text = str(args.get("top_vendor", "")).lower()

    has_all_ids = all(tid in ids_text for tid in _ARGTRANS_REQUIRED_IDS_STATEFUL)
    has_total = any(tok in total_text for tok in _ARGTRANS_TOTAL_TOKENS_STATEFUL)
    has_vendor = _ARGTRANS_TOP_VENDOR_STATEFUL in vendor_text
    return has_all_ids and has_total and has_vendor


def _build_argument_transformation_stateful() -> tuple[Workflow, callable]:
    db = ExpenseAuditSystem()
    tools: dict[str, ToolDef] = {
        "list_transactions": ToolDef(
            spec=ToolSpec(
                name="list_transactions",
                description="List all expense transactions for a given fiscal quarter and year.",
                parameters=QuarterYearParams,
            ),
            callable=lambda **kw: db.list_transactions(kw["quarter"], kw["year"]),
        ),
        "get_approved_vendors": ToolDef(
            spec=ToolSpec(
                name="get_approved_vendors",
                description="Return the canonical list of approved vendor names (case sensitive).",
                parameters=NoParams,
            ),
            callable=lambda **kw: db.get_approved_vendors(),
        ),
        "get_vendor_details": ToolDef(
            spec=ToolSpec(
                name="get_vendor_details",
                description="Look up vendor details (status, legal entity, trade-name aliases).",
                parameters=VendorNameParams,
            ),
            callable=lambda **kw: db.get_vendor_details(kw["vendor_name"]),
        ),
        "currency_convert": ToolDef(
            spec=ToolSpec(
                name="currency_convert",
                description="Convert an amount between currencies (USD/EUR).",
                parameters=CurrencyConvertParams,
            ),
            callable=lambda **kw: db.currency_convert(
                kw["amount"], kw["from_currency"], kw["to_currency"],
            ),
        ),
        "categorize_expense": ToolDef(
            spec=ToolSpec(
                name="categorize_expense",
                description="Assign an expense to a GL category bucket.",
                parameters=CategorizeExpenseParams,
            ),
            callable=lambda **kw: db.categorize_expense(kw["amount"], kw["category"]),
        ),
        "lookup_transaction": ToolDef(
            spec=ToolSpec(
                name="lookup_transaction",
                description="Look up a single transaction's full record by ID.",
                parameters=TransactionIdParams,
            ),
            callable=lambda **kw: db.lookup_transaction(kw["transaction_id"]),
        ),
        "submit_audit_report": ToolDef(
            spec=ToolSpec(
                name="submit_audit_report",
                description=(
                    "Submit the completed audit report. Provide comma-separated "
                    "transaction IDs of flagged items, the total flagged amount in "
                    "USD, and the vendor of the single largest flagged transaction."
                ),
                parameters=SubmitAuditReportParams,
            ),
            callable=lambda **kw: db.submit_audit_report(
                kw["transaction_ids"], kw["total_flagged_usd"], kw["top_vendor"],
            ),
        ),
    }
    workflow = Workflow(
        name="argument_transformation_stateful",
        description=(
            "Run a Q4 expense audit by deriving the flagged-transaction "
            "set, total in USD, and top vendor from heterogeneous tool "
            "data, then submitting the audit report."
        ),
        tools=tools,
        required_steps=["list_transactions", "get_approved_vendors"],
        terminal_tool="submit_audit_report",
        system_prompt_template=(
            "You are an expense audit assistant. Use the available tools "
            "to identify flagged transactions and submit a complete audit "
            "report. Read tool results carefully — amounts may be in "
            "different currencies and vendor records may have aliases."
        ),
    )
    # State validator: model must have used currency_convert for the EUR
    # transaction AND looked up vendor details for the ACME Corp alias
    # AND submitted the canonical-correct args.
    validate_state = lambda: (
        db.list_called_for == ("Q4", 2024)
        and db.approved_called
        and db.eur_conversion_called
        and "ACME Corp" in db.vendor_details_called_for
        and db.submitted_args is not None
        and _validate_argument_transformation_stateful(db.submitted_args)
    )
    return workflow, validate_state


argument_transformation_stateful = EvalScenario(
    name="argument_transformation_stateful",
    description=(
        "Stateful argument transformation — filter + currency-convert + "
        "vendor disambiguation + aggregate; state tracks whether the "
        "model used currency_convert and get_vendor_details."
    ),
    workflow=_placeholder_workflow(
        "argument_transformation_stateful", "submit_audit_report",
        ["list_transactions", "get_approved_vendors"],
    ),
    user_message=(
        "Run our Q4 2024 expense audit. Flag any transaction of $5,000 "
        "or more from vendors NOT on our approved list. Submit the audit "
        "report with: comma-separated transaction IDs, total flagged "
        "amount in USD, and the vendor of the single largest flagged "
        "transaction."
    ),
    validate=_validate_argument_transformation_stateful,
    build_workflow=_build_argument_transformation_stateful,
    tags=["stateful", "advanced_reasoning", "reasoning", "model_quality"],
    ideal_iterations=5,
    max_iterations=15,
)
