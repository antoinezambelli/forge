"""Advanced model reasoning scenarios — designed to weed out top-tier
models after the basic model_quality suite saturated post-sampling-fix.

These scenarios layer noise, traps, and multi-hop derivations on top of
the basic capability tests. They share the indirect-validation pattern
(substring-AND on terminal args; downstream tools encode whether the
model derived the right intermediates).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario
from ._model_quality import (
    EmployeeIdParams,
    EmployeeNameParams,
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


# ── Scenario: data_gap_recovery_extended ────────────────────────
#
# Extends data_gap_recovery: 5 facts to assemble (vs 3), 2-3 hop chains
# with NO direct tool-name hints (model must discover tools by name and
# infer routing from domain language), 2 misleading-by-name traps
# (compensation_history, access_control), and 1 status-marker lure
# (legacy_records — plausible-but-stale data with an embedded redirect
# note). Plus a soft trap: security_audit returns provisioned access
# groups (payments-prod, internal-apis) that look like a complete answer
# but are missing the v3/v2 suffixes — the model must realize these are
# stale and find project_assignments to get the current versions.
# Same indirect-validation pattern as DGR — the unique markers in the
# report (B7, payments-prod-v3, internal-apis-v2) only appear if the
# model reached the right downstream tools.

_data_gap_recovery_extended_tools: dict[str, ToolDef] = {
    "get_employee": ToolDef(
        spec=ToolSpec(
            name="get_employee",
            description="Look up an employee record by name.",
            parameters=EmployeeNameParams,
        ),
        callable=lambda **kwargs: (
            "Employee Record — Sarah Chen (ID: E-1847)\n"
            "Department: Engineering\n"
            "Title: Senior Backend Engineer\n"
            "Start Date: 2019-03-15\n"
            "Office: Building 3, Floor 2\n"
            "Manager: David Park\n"
            "Clearance Level: [RESTRICTED]\n"
            "Emergency Contact: [unavailable in this record]\n"
            "Salary Band: [PROTECTED]\n"
            "Access Groups: [managed externally]"
        ),
    ),
    "security_audit": ToolDef(
        spec=ToolSpec(
            name="security_audit",
            description="Query security audit records for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Security Audit Record — E-1847 (Sarah Chen)\n"
            "Clearance Level: L3 — Confidential\n"
            "Granted: 2021-06-10\n"
            "Sponsor: David Park (Director, Engineering)\n"
            "Last Review: 2024-12-01 (passed, no findings)\n"
            "Expires: 2025-12-01\n"
            "Access Groups: payments-prod, internal-apis, staging-*"
        ),
    ),
    "onboarding_records": ToolDef(
        spec=ToolSpec(
            name="onboarding_records",
            description="Access HR onboarding records for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "HR Onboarding Record — E-1847 (Sarah Chen)\n"
            "Original Onboarding Date: 2019-03-15\n"
            "Emergency Contact: Michael Chen (spouse) — (555) 867-5309\n"
            "Dietary Restrictions: None\n"
            "T-Shirt Size: M\n"
            'Equipment Issued: MacBook Pro 16", 2x monitors'
        ),
    ),
    "personnel_file": ToolDef(
        spec=ToolSpec(
            name="personnel_file",
            description="Access an employee's personnel file.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Personnel File — E-1847 (Sarah Chen)\n"
            "Contains: performance reviews, PTO balance.\n"
            "Note: Compensation data is in our current pay system (migrated 2024).\n"
            "Note: Emergency contact records are with HR onboarding (2020 migration)."
        ),
    ),
    "compensation_v2": ToolDef(
        spec=ToolSpec(
            name="compensation_v2",
            description="Current compensation system — salary band lookup.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Compensation v2 — E-1847 (Sarah Chen)\n"
            "Salary Band: B7\n"
            "Effective: 2024-04-01\n"
            "Next Review: 2025-04-01"
        ),
    ),
    "project_assignments": ToolDef(
        spec=ToolSpec(
            name="project_assignments",
            description="Current project assignments and active access groups for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Project Assignments — E-1847 (Sarah Chen)\n"
            "Active Access Groups: payments-prod-v3, internal-apis-v2, staging-test\n"
            "Primary Project: Payments Platform Modernization\n"
            "Secondary: Internal Tooling Sprint"
        ),
    ),
    "compensation_history": ToolDef(
        spec=ToolSpec(
            name="compensation_history",
            description="Look up compensation history for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "compensation_history — DEPRECATED endpoint.\n"
            "This service was retired in 2024. Use compensation_v2 for "
            "current salary band data."
        ),
    ),
    "access_control": ToolDef(
        spec=ToolSpec(
            name="access_control",
            description="Look up access control entries for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "access_control — DEPRECATED endpoint.\n"
            "Replaced by project_assignments for current active access "
            "group data. Static provisioning records remain in security_audit."
        ),
    ),
    "legacy_records": ToolDef(
        spec=ToolSpec(
            name="legacy_records",
            description="Search legacy records archive for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Legacy Archive — E-1847 (Sarah Chen)\n"
            "Source: legacy_v1 (READ-ONLY — not synced with current systems)\n"
            "Salary Band: B5\n"
            "Access Groups: legacy-readonly\n"
            "Note: legacy_v1 stopped syncing after the 2024 systems migration. "
            "For current salary band, query compensation_v2. For current access "
            "groups, query project_assignments. Data above is provided for "
            "historical audit only."
        ),
    ),
    "compliance_check": ToolDef(
        spec=ToolSpec(
            name="compliance_check",
            description="Check compliance status for an employee.",
            parameters=EmployeeIdParams,
        ),
        callable=lambda **kwargs: (
            "Compliance Status — E-1847: All mandatory trainings complete.\n"
            "Last security awareness training: 2024-11-15.\n"
            "No outstanding compliance items."
        ),
    ),
    "hr_directory": ToolDef(
        spec=ToolSpec(
            name="hr_directory",
            description="Look up an employee in the HR directory.",
            parameters=EmployeeNameParams,
        ),
        callable=lambda **kwargs: (
            "HR Directory Entry — Sarah Chen\n"
            "Department: Engineering | Reports to: David Park\n"
            "Office: Building 3, Floor 2 | Extension: x4481\n"
            "Status: Active | Full-time"
        ),
    ),
    "submit_report": ToolDef(
        spec=ToolSpec(
            name="submit_report",
            description="Submit a completed employee profile report.",
            parameters=SubmitReportParams,
        ),
        callable=lambda **kwargs: f"Report submitted for {kwargs.get('employee_name', '')}.",
    ),
}


def _validate_data_gap_recovery_extended(args: dict[str, Any]) -> bool:
    text = f"{args.get('employee_name', '')} {args.get('report', '')}".lower()
    has_dept      = "engineering" in text
    has_clearance = "l3" in text or "confidential" in text
    has_contact   = "michael" in text and ("867-5309" in text or "spouse" in text)
    has_salary    = "b7" in text  # only in compensation_v2; "b5" is the legacy_records lure
    has_groups    = "payments-prod-v3" in text or "internal-apis-v2" in text  # only in project_assignments
    return has_dept and has_clearance and has_contact and has_salary and has_groups


data_gap_recovery_extended = EvalScenario(
    name="data_gap_recovery_extended",
    description="Extended hint-chain — 5 facts, 2-3 hop chains, no direct hints, name-trap + status-marker + soft-trap lures.",
    workflow=Workflow(
        name="data_gap_recovery_extended",
        description=(
            "Build a complete employee profile with multi-hop data resolution "
            "across legacy and current systems."
        ),
        tools=_data_gap_recovery_extended_tools,
        required_steps=["get_employee"],
        terminal_tool="submit_report",
        system_prompt_template=(
            "You are an HR systems assistant. Use the available tools to "
            "gather the requested employee information and submit a complete "
            "profile report. Read tool results carefully — some endpoints "
            "redirect to newer systems."
        ),
    ),
    user_message=(
        "Pull together a complete profile for Sarah Chen — we need her "
        "clearance level, current emergency contact, latest salary band, "
        "and active access groups for the onboarding audit and access review."
    ),
    validate=_validate_data_gap_recovery_extended,
    tags=["advanced_reasoning", "reasoning", "model_quality"],
    ideal_iterations=8,
    max_iterations=20,
)


# ── Scenario: argument_transformation ───────────────────────────
#
# Tests reasoning over data, not symbolic plumbing. Tools return
# heterogeneous structured data (transactions in mixed currencies,
# approved-vendor list, vendor aliases); the model must filter,
# convert, disambiguate, and aggregate before calling the terminal
# tool with the derived results.
#
# Top-tier separators (designed to weed out 27B+ models, like DGE):
#  1. Currency conversion: EUR transactions need conversion to USD
#     before applying the $5K threshold. Skipping -> wrong inclusion
#     set + wrong total.
#  2. Vendor case-mismatch: 'ACME Corp' is a registered alias of the
#     approved 'Acme Corp'. Literal string compare over-flags it;
#     get_vendor_details resolves the alias.
#  3. Threshold boundary: '$5,000 or more' -> >=, not >. The $5,000
#     transaction must be included.
#
# Validation pattern (same as DGE / DGR): substring-AND on terminal
# args. Correct filter -> correct ID set + total + top vendor ->
# substring matches succeed. Any reasoning slip -> mismatch -> fail.

# Fixed FX rate for determinism (no live currency lookup).
_FX_EUR_TO_USD = 1.10

_APPROVED_VENDORS = [
    "Acme Corp",
    "Globex Industries",
    "Initech Systems",
    "Umbrella Logistics",
    "Wayne Enterprises",
    "Stark Industries",
]

# Q4 2024 transactions. The 'flag' column is the ground-truth label
# for unit testing; it's not exposed to the model.
_Q4_2024_TRANSACTIONS = [
    # id        date          vendor                amount  currency  flag
    ("TX-1001", "2024-10-05", "Cyberdyne LLC",       7500,  "USD",    True),   # unapproved, >=$5K
    ("TX-1002", "2024-10-12", "Acme Corp",          12000,  "USD",    False),  # approved
    ("TX-1003", "2024-10-22", "Initech Systems",     8200,  "USD",    False),  # approved
    ("TX-1004", "2024-11-03", "Vandelay Imports",    3500,  "USD",    False),  # under $5K
    ("TX-1005", "2024-11-08", "Soylent Corp",        5000,  "USD",    True),   # boundary: $5K exactly
    ("TX-1006", "2024-11-14", "Pied Piper",          4800,  "EUR",    True),   # =$5,280 USD, unapproved
    ("TX-1007", "2024-11-22", "Umbrella Logistics",  9400,  "USD",    False),  # approved
    ("TX-1008", "2024-12-02", "Wonka Industries",   11200,  "USD",    True),   # unapproved, >=$5K, top
    ("TX-1009", "2024-12-08", "ACME Corp",           6500,  "USD",    False),  # case-mismatch alias
    ("TX-1010", "2024-12-15", "Globex Industries",   5500,  "USD",    False),  # approved
    ("TX-1011", "2024-12-19", "Pied Piper",          2400,  "EUR",    False),  # =$2,640 USD, under $5K
    ("TX-1012", "2024-12-22", "Stark Industries",   14800,  "USD",    False),  # approved
    ("TX-1013", "2024-12-28", "Wayne Enterprises",   7300,  "USD",    False),  # approved
]


def _format_transactions(rows: list[tuple[str, str, str, int, str, bool]]) -> str:
    lines = ["Q4 2024 Expense Transactions:"]
    for tid, date, vendor, amount, cur, _flag in rows:
        lines.append(
            f"  {tid} | {date} | vendor: {vendor:25s} | "
            f"amount: {amount:>8,.2f} {cur}"
        )
    return "\n".join(lines)


def _list_transactions(**kwargs: Any) -> str:
    quarter = str(kwargs.get("quarter", "")).strip().upper()
    year = int(kwargs.get("year", 0))
    if quarter == "Q4" and year == 2024:
        return _format_transactions(_Q4_2024_TRANSACTIONS)
    return f"No transactions found for {quarter} {year}."


def _get_approved_vendors(**kwargs: Any) -> str:
    lines = ["Approved Vendors (canonical names — case sensitive):"]
    for v in _APPROVED_VENDORS:
        lines.append(f"  - {v}")
    return "\n".join(lines)


def _get_vendor_details(**kwargs: Any) -> str:
    name = str(kwargs.get("vendor_name", "")).strip()
    # The case-mismatch trap: ACME Corp is the registered alias of Acme Corp.
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
    # Other approved vendors return generic stubs.
    if name in _APPROVED_VENDORS:
        return f"Vendor Details — {name}\nStatus: standard supplier (active)."
    # Unknown / unapproved vendors return a no-record stub. No useful info,
    # but doesn't break the model — it can still flag based on the approved
    # list directly.
    return f"Vendor Details — {name}\nStatus: not found in vendor master."


def _currency_convert(**kwargs: Any) -> str:
    amount = float(kwargs.get("amount", 0))
    src = str(kwargs.get("from_currency", "")).strip().upper()
    dst = str(kwargs.get("to_currency", "")).strip().upper()
    if src == "EUR" and dst == "USD":
        converted = amount * _FX_EUR_TO_USD
        return (
            f"Conversion: {amount:,.2f} EUR = {converted:,.2f} USD "
            f"(rate: 1 EUR = {_FX_EUR_TO_USD} USD)"
        )
    if src == "USD" and dst == "EUR":
        converted = amount / _FX_EUR_TO_USD
        return (
            f"Conversion: {amount:,.2f} USD = {converted:,.2f} EUR "
            f"(rate: 1 EUR = {_FX_EUR_TO_USD} USD)"
        )
    if src == dst:
        return f"Conversion: {amount:,.2f} {src} = {amount:,.2f} {dst} (same currency)."
    return f"Unsupported conversion pair: {src} -> {dst}."


def _categorize_expense(**kwargs: Any) -> str:
    # Distractor: returns a category bucket, no useful audit info.
    amount = float(kwargs.get("amount", 0))
    category = str(kwargs.get("category", "")).strip()
    return (
        f"Categorization: {amount:,.2f} -> {category} "
        f"(GL bucket: GL-{abs(hash(category)) % 9000 + 1000})."
    )


def _lookup_transaction(**kwargs: Any) -> str:
    # Redundant helper: returns a single transaction's record. Distractor —
    # list_transactions already returns everything.
    tid = str(kwargs.get("transaction_id", "")).strip()
    for row in _Q4_2024_TRANSACTIONS:
        if row[0] == tid:
            _id, date, vendor, amount, cur, _flag = row
            return (
                f"Transaction Record — {tid}\n"
                f"Date: {date}\nVendor: {vendor}\n"
                f"Amount: {amount:,.2f} {cur}"
            )
    return f"No transaction found for '{tid}'."


_argument_transformation_tools: dict[str, ToolDef] = {
    "list_transactions": ToolDef(
        spec=ToolSpec(
            name="list_transactions",
            description="List all expense transactions for a given fiscal quarter and year.",
            parameters=QuarterYearParams,
        ),
        callable=_list_transactions,
    ),
    "get_approved_vendors": ToolDef(
        spec=ToolSpec(
            name="get_approved_vendors",
            description="Return the canonical list of approved vendor names (case sensitive).",
            parameters=NoParams,
        ),
        callable=_get_approved_vendors,
    ),
    "get_vendor_details": ToolDef(
        spec=ToolSpec(
            name="get_vendor_details",
            description="Look up vendor details (status, legal entity, trade-name aliases).",
            parameters=VendorNameParams,
        ),
        callable=_get_vendor_details,
    ),
    "currency_convert": ToolDef(
        spec=ToolSpec(
            name="currency_convert",
            description="Convert an amount between currencies (USD/EUR).",
            parameters=CurrencyConvertParams,
        ),
        callable=_currency_convert,
    ),
    "categorize_expense": ToolDef(
        spec=ToolSpec(
            name="categorize_expense",
            description="Assign an expense to a GL category bucket.",
            parameters=CategorizeExpenseParams,
        ),
        callable=_categorize_expense,
    ),
    "lookup_transaction": ToolDef(
        spec=ToolSpec(
            name="lookup_transaction",
            description="Look up a single transaction's full record by ID.",
            parameters=TransactionIdParams,
        ),
        callable=_lookup_transaction,
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
        callable=lambda **kwargs: (
            f"Audit report submitted. "
            f"Flagged: {kwargs.get('transaction_ids', '')}; "
            f"total: {kwargs.get('total_flagged_usd', '')}; "
            f"top: {kwargs.get('top_vendor', '')}."
        ),
    ),
}


# Expected ground truth (used only for validation):
#  Flagged: TX-1001 ($7,500), TX-1005 ($5,000), TX-1006 ($5,280 from €4,800),
#           TX-1008 ($11,200) -> total $28,980; top vendor Wonka Industries.
_ARGTRANS_REQUIRED_IDS = ("TX-1001", "TX-1005", "TX-1006", "TX-1008")
_ARGTRANS_TOTAL_TOKENS = ("28,980", "28980")  # accept either format
_ARGTRANS_TOP_VENDOR = "wonka"


def _validate_argument_transformation(args: dict[str, Any]) -> bool:
    ids_text = str(args.get("transaction_ids", ""))
    total_text = str(args.get("total_flagged_usd", "")).replace("$", "")
    vendor_text = str(args.get("top_vendor", "")).lower()

    has_all_ids = all(tid in ids_text for tid in _ARGTRANS_REQUIRED_IDS)
    has_total = any(tok in total_text for tok in _ARGTRANS_TOTAL_TOKENS)
    has_vendor = _ARGTRANS_TOP_VENDOR in vendor_text
    return has_all_ids and has_total and has_vendor


argument_transformation = EvalScenario(
    name="argument_transformation",
    description=(
        "Argument transformation — filter + currency-convert + vendor "
        "disambiguation + aggregate over structured tool returns."
    ),
    workflow=Workflow(
        name="argument_transformation",
        description=(
            "Run a Q4 expense audit by deriving the flagged-transaction "
            "set, total in USD, and top vendor from heterogeneous tool "
            "data, then submitting the audit report."
        ),
        tools=_argument_transformation_tools,
        required_steps=["list_transactions", "get_approved_vendors"],
        terminal_tool="submit_audit_report",
        system_prompt_template=(
            "You are an expense audit assistant. Use the available tools "
            "to identify flagged transactions and submit a complete audit "
            "report. Read tool results carefully — amounts may be in "
            "different currencies and vendor records may have aliases."
        ),
    ),
    user_message=(
        "Run our Q4 2024 expense audit. Flag any transaction of $5,000 "
        "or more from vendors NOT on our approved list. Submit the audit "
        "report with: comma-separated transaction IDs, total flagged "
        "amount in USD, and the vendor of the single largest flagged "
        "transaction."
    ),
    validate=_validate_argument_transformation,
    tags=["advanced_reasoning", "reasoning", "model_quality"],
    ideal_iterations=5,
    max_iterations=15,
)
