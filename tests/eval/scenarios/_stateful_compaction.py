"""Stateful compaction scenarios — compaction stress, phase2, inventory audit,
supplier deep dive, relevance detection."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from forge.core.workflow import ToolDef, ToolSpec, Workflow

from ._base import EvalScenario, _check, _placeholder_workflow
from ._compaction import _SUPPLIER_DATA, _validate_phase2_compaction
from ._plumbing import _COMPACTION_ANALYZE_RESULT, _COMPACTION_FETCH_RESULT


class FetchSalesParams(BaseModel):
    quarter: int = Field(description="Quarter number (1-4)")
    year: int = Field(description="Four-digit year")


class EmptyParams(BaseModel):
    pass


class FindingsParams(BaseModel):
    findings: str = Field(description="The findings to include in the report")


class SupplierParams(BaseModel):
    supplier: str = Field(description="The supplier name to check")


class RecommendParams(BaseModel):
    primary_supplier: str = Field(description="The recommended primary supplier name")
    reasoning: str = Field(description="Justification for the recommendation, including key metrics")


class SectionParams(BaseModel):
    section_id: str = Field(description="The warehouse section ID to scan (e.g. 'A', 'B', ...)")


class ItemParams(BaseModel):
    item_id: str = Field(description="The item ID to investigate (from a prior scan)")


class AuditReportParams(BaseModel):
    findings: str = Field(description="Summary of audit findings including discrepancies")


class ComponentParams(BaseModel):
    component: str = Field(description="The component name to search for")


class SupplierNameParams(BaseModel):
    supplier: str = Field(description="The supplier name")


class SupplierRecommendParams(BaseModel):
    supplier: str = Field(description="The recommended supplier name")
    reasoning: str = Field(
        description="Justification citing price, quality, and reference metrics",
    )


class CityParams(BaseModel):
    city: str = Field(description="The city name")


class FlightParams(BaseModel):
    origin: str = Field(description="Departure city")
    destination: str = Field(description="Arrival city")


class HotelParams(BaseModel):
    city: str = Field(description="The city to check")
    checkin: str = Field(description="Check-in date")


class CurrencyParams(BaseModel):
    amount: str = Field(description="The amount to convert")
    from_currency: str = Field(description="Source currency code")
    to_currency: str = Field(description="Target currency code")


class ReasonParams(BaseModel):
    reason: str = Field(description="Brief explanation of why no tool is appropriate")


# ── Backend 10: SalesReportPipeline ────────────────────────────


class SalesReportPipeline:
    """Stateful backend for compaction_stress — large tool results under tight budget."""

    def __init__(self) -> None:
        self.loaded_data: str | None = None
        self.analysis: str | None = None

    def fetch_sales_data(self, quarter: int, year: int) -> str:
        if (quarter, year) != (4, 2024):
            return f"No data found for Q{quarter} {year}."
        self.loaded_data = _COMPACTION_FETCH_RESULT
        return self.loaded_data

    def analyze_sales(self) -> str:
        if self.loaded_data is None:
            return "Error: no data loaded. Call fetch_sales_data first."
        self.analysis = _COMPACTION_ANALYZE_RESULT
        return self.analysis

    def report(self, findings: str) -> str:
        return findings  # echo-back terminal


def _build_compaction_stress_stateful() -> tuple[Workflow, callable]:
    db = SalesReportPipeline()
    tools: dict[str, ToolDef] = {
        "fetch_sales_data": ToolDef(
            spec=ToolSpec(
                name="fetch_sales_data",
                description="Fetch sales data for a given quarter and year.",
                parameters=FetchSalesParams,
            ),
            callable=lambda **kw: db.fetch_sales_data(kw["quarter"], kw["year"]),
        ),
        "analyze_sales": ToolDef(
            spec=ToolSpec(
                name="analyze_sales",
                description="Analyze the loaded sales data and produce findings.",
                parameters=EmptyParams,
            ),
            callable=lambda **kw: db.analyze_sales(),
        ),
        "report": ToolDef(
            spec=ToolSpec(
                name="report",
                description="Produce a final report.",
                parameters=FindingsParams,
            ),
            callable=lambda **kw: db.report(kw.get("findings", "")),
        ),
    }
    workflow = Workflow(
        name="compaction_stress_stateful",
        description="Fetch, analyze, report under tight token budget",
        tools=tools,
        required_steps=["fetch_sales_data", "analyze_sales"],
        terminal_tool="report",
        system_prompt_template=(
            "You are a data analyst assistant. Fetch the sales data first, "
            "then analyze it, then produce a report using the report tool."
        ),
    )
    validate_state = lambda: (
        db.loaded_data is not None
        and db.analysis is not None
    )
    return workflow, validate_state


compaction_stress_stateful = EvalScenario(
    name="compaction_stress_stateful",
    description="Stateful compaction stress — large tool results with argument-dependent loading.",
    workflow=_placeholder_workflow(
        "compaction_stress_stateful", "report",
        ["fetch_sales_data", "analyze_sales"],
    ),
    user_message="Generate a sales report from the Q4 2024 dataset.",
    budget_tokens=2048,
    validate=lambda args: _check(args.get("findings", ""), ["23", "widget pro", "west"]),
    build_workflow=_build_compaction_stress_stateful,
    tags=["stateful", "compaction"],
    ideal_iterations=3,
)


# ── Backend 11: SupplierAuditSystem ───────────────────────────


class SupplierAuditSystem:
    """Stateful backend for phase2_compaction — tracks which suppliers were checked."""

    def __init__(self) -> None:
        self.suppliers_checked: list[str] = []

    def check_supplier(self, supplier: str) -> str:
        name = supplier.strip().lower()
        for key in _SUPPLIER_DATA:
            if key in name or name in key:
                self.suppliers_checked.append(key)
                return _SUPPLIER_DATA[key]
        return (
            f"Unknown supplier: '{supplier}'. "
            f"Available suppliers: AlphaTech, Meridian, NovaCore, PrecisionWorks, CircuitEdge."
        )

    def recommend(self, primary_supplier: str, reasoning: str) -> str:
        return f"Recommendation recorded: {primary_supplier}"


def _build_phase2_compaction_stateful() -> tuple[Workflow, callable]:
    db = SupplierAuditSystem()
    tools: dict[str, ToolDef] = {
        "check_supplier": ToolDef(
            spec=ToolSpec(
                name="check_supplier",
                description="Check a supplier's assessment data for the MCU-4200 component.",
                parameters=SupplierParams,
            ),
            callable=lambda **kw: db.check_supplier(kw["supplier"]),
        ),
        "recommend": ToolDef(
            spec=ToolSpec(
                name="recommend",
                description="Submit a procurement recommendation based on supplier assessments.",
                parameters=RecommendParams,
            ),
            callable=lambda **kw: db.recommend(
                kw.get("primary_supplier", ""), kw.get("reasoning", ""),
            ),
        ),
    }
    workflow = Workflow(
        name="phase2_compaction_stateful",
        description="Check 5 suppliers for MCU-4200 component, then recommend the best one",
        tools=tools,
        required_steps=["check_supplier"],
        terminal_tool="recommend",
        system_prompt_template=(
            "You are a procurement analyst. Evaluate all five suppliers for "
            "the MCU-4200 microcontroller by checking each one, then submit a "
            "recommendation for the primary supplier. Check suppliers in this "
            "order: AlphaTech, Meridian, NovaCore, PrecisionWorks, CircuitEdge. "
            "Your recommendation must name the best supplier and cite specific "
            "metrics (price, quality, capacity) that justify the choice."
        ),
    )
    validate_state = lambda: (
        len(db.suppliers_checked) >= 4
        and "precisionworks" in db.suppliers_checked
    )
    return workflow, validate_state


phase2_compaction_stateful = EvalScenario(
    name="phase2_compaction_stateful",
    description="Stateful P2 compaction — verifies suppliers were actually checked under tight budget.",
    workflow=_placeholder_workflow(
        "phase2_compaction_stateful", "recommend", ["check_supplier"],
    ),
    user_message=(
        "We need to source 10,000 MCU-4200 microcontrollers. Evaluate all "
        "five suppliers and recommend the best primary supplier."
    ),
    budget_tokens=925,
    validate=_validate_phase2_compaction,
    build_workflow=_build_phase2_compaction_stateful,
    tags=["stateful", "compaction", "reasoning"],
    ideal_iterations=6,
)


# ── Backend 12: WarehouseSystem (inventory_audit) ──────────────


# 8 sections, 5-8 items each. Some items flagged with discrepancies.
# Tool results ~250 chars per scan, ~200 chars per investigation.
_WAREHOUSE_SECTIONS: dict[str, list[dict[str, Any]]] = {
    "A": [
        {"id": "A-001", "name": "Hex Bolts M8x30", "qty": 1200, "expected": 1200},
        {"id": "A-002", "name": "Flat Washers M8", "qty": 2400, "expected": 2400},
        {"id": "A-003", "name": "Lock Nuts M8", "qty": 890, "expected": 1000,
         "flag": "shortage", "detail": "110 units short. Last shipment 2025-01-12 was 110 units light per receiving log. Supplier: FastenCo, PO #FC-8812."},
        {"id": "A-004", "name": "Spring Pins 4x20", "qty": 3100, "expected": 3100},
        {"id": "A-005", "name": "Cotter Pins 3mm", "qty": 5600, "expected": 5600},
    ],
    "B": [
        {"id": "B-001", "name": "Bearing 6205-2RS", "qty": 340, "expected": 340},
        {"id": "B-002", "name": "Bearing 6308-ZZ", "qty": 180, "expected": 180},
        {"id": "B-003", "name": "O-Ring Kit AS568", "qty": 420, "expected": 500,
         "flag": "shortage", "detail": "80 units short. Kits consumed faster than forecast — line 3 using 2x rate since Dec retrofit. Reorder needed."},
        {"id": "B-004", "name": "Shaft Collar 25mm", "qty": 200, "expected": 200},
        {"id": "B-005", "name": "Linear Rail HGR15", "qty": 48, "expected": 48},
        {"id": "B-006", "name": "Ball Screw SFU1605", "qty": 24, "expected": 24},
    ],
    "C": [
        {"id": "C-001", "name": "Aluminum Sheet 2mm", "qty": 75, "expected": 75},
        {"id": "C-002", "name": "Steel Plate 5mm", "qty": 42, "expected": 42},
        {"id": "C-003", "name": "Copper Bar 12mm", "qty": 30, "expected": 30},
        {"id": "C-004", "name": "Brass Rod 8mm", "qty": 65, "expected": 65},
        {"id": "C-005", "name": "Nylon Block 50mm", "qty": 110, "expected": 110},
    ],
    "D": [
        {"id": "D-001", "name": "Servo Motor NEMA23", "qty": 18, "expected": 18},
        {"id": "D-002", "name": "Stepper Driver DM542", "qty": 22, "expected": 22},
        {"id": "D-003", "name": "Power Supply 24V 15A", "qty": 14, "expected": 15,
         "flag": "shortage", "detail": "1 unit short. Found in Section F during cross-check — misplaced after maintenance on 2025-02-01. Item physically intact."},
        {"id": "D-004", "name": "Relay Module 8ch", "qty": 30, "expected": 30},
        {"id": "D-005", "name": "PLC Siemens S7-1200", "qty": 6, "expected": 6},
        {"id": "D-006", "name": "HMI Panel 7in", "qty": 8, "expected": 8},
        {"id": "D-007", "name": "Cable Tray 2m", "qty": 45, "expected": 45},
    ],
    "E": [
        {"id": "E-001", "name": "Hydraulic Hose 1/2in", "qty": 200, "expected": 200},
        {"id": "E-002", "name": "Pneumatic Fitting 6mm", "qty": 800, "expected": 800},
        {"id": "E-003", "name": "Pressure Gauge 0-10bar", "qty": 35, "expected": 35},
        {"id": "E-004", "name": "Solenoid Valve 24V", "qty": 42, "expected": 42},
        {"id": "E-005", "name": "Air Filter Element", "qty": 60, "expected": 60},
    ],
    "F": [
        {"id": "F-001", "name": "Safety Gloves XL", "qty": 150, "expected": 150},
        {"id": "F-002", "name": "Safety Goggles", "qty": 85, "expected": 80,
         "flag": "surplus", "detail": "5 extra units. Duplicate delivery on 2025-01-20 — supplier sent replacement before RMA was processed. Credit memo pending, ref #SG-4401."},
        {"id": "F-003", "name": "Ear Protection NRR30", "qty": 120, "expected": 120},
        {"id": "F-004", "name": "First Aid Kit", "qty": 12, "expected": 12},
        {"id": "F-005", "name": "Fire Extinguisher 5kg", "qty": 20, "expected": 20},
        {"id": "F-006", "name": "Spill Kit Universal", "qty": 8, "expected": 8},
    ],
    "G": [
        {"id": "G-001", "name": "Cutting Fluid 20L", "qty": 24, "expected": 24},
        {"id": "G-002", "name": "Lubricant Grease 1kg", "qty": 36, "expected": 36},
        {"id": "G-003", "name": "Thread Sealant 50ml", "qty": 48, "expected": 48},
        {"id": "G-004", "name": "Degreaser Spray 400ml", "qty": 72, "expected": 72},
        {"id": "G-005", "name": "Rust Inhibitor 1L", "qty": 18, "expected": 18},
    ],
    "H": [
        {"id": "H-001", "name": "Welding Rod E7018 3.2mm", "qty": 500, "expected": 500},
        {"id": "H-002", "name": "MIG Wire ER70S-6 1mm", "qty": 40, "expected": 40},
        {"id": "H-003", "name": "Grinding Disc 125mm", "qty": 200, "expected": 200},
        {"id": "H-004", "name": "Cut-off Wheel 230mm", "qty": 95, "expected": 100,
         "flag": "expired", "detail": "5 wheels past expiry date 2024-12-31. Manufacturer lot #GD-7790. Must be disposed per safety policy — expired abrasives risk fracture under load."},
        {"id": "H-005", "name": "Flap Disc 115mm", "qty": 60, "expected": 60},
        {"id": "H-006", "name": "TIG Tungsten 2.4mm", "qty": 30, "expected": 30},
    ],
}

# Items with flags — the model should investigate these after scanning.
_FLAGGED_ITEMS: dict[str, dict[str, Any]] = {}
for _sec_id, _items in _WAREHOUSE_SECTIONS.items():
    for _item in _items:
        if "flag" in _item:
            _FLAGGED_ITEMS[_item["id"].lower()] = _item


class WarehouseSystem:
    """Stateful backend for inventory_audit — 8 sections, flagged item investigation."""

    def __init__(self) -> None:
        self.scanned: list[str] = []
        self.investigated: list[str] = []

    def scan_section(self, section_id: str) -> str:
        sid = section_id.strip().upper()
        if sid not in _WAREHOUSE_SECTIONS:
            return (
                f"Unknown section: '{section_id}'. "
                f"Valid sections: {', '.join(sorted(_WAREHOUSE_SECTIONS.keys()))}."
            )
        if sid not in self.scanned:
            self.scanned.append(sid)
        items = _WAREHOUSE_SECTIONS[sid]
        lines = [f"Section {sid} — {len(items)} items:"]
        for it in items:
            status = "OK"
            if "flag" in it:
                status = f"FLAG:{it['flag']} (qty {it['qty']}/{it['expected']})"
            lines.append(f"  {it['id']} {it['name']}: {it['qty']} — {status}")
        return "\n".join(lines)

    def check_item(self, item_id: str) -> str:
        iid = item_id.strip().lower()
        # Must have scanned the section first
        if iid in _FLAGGED_ITEMS:
            section = iid.split("-")[0].upper()
            if section not in self.scanned:
                return (
                    f"Error: section {section} has not been scanned yet. "
                    f"Call scan_section('{section}') first."
                )
            if iid not in self.investigated:
                self.investigated.append(iid)
            item = _FLAGGED_ITEMS[iid]
            return (
                f"Investigation — {item['id']} {item['name']}:\n"
                f"  Type: {item['flag']} | Expected: {item['expected']} | Actual: {item['qty']}\n"
                f"  Detail: {item['detail']}"
            )
        return f"Item '{item_id}' is not flagged or does not exist."

    def file_report(self, findings: str) -> str:
        return findings  # echo-back terminal


def _build_inventory_audit() -> tuple[Workflow, callable]:
    db = WarehouseSystem()
    tools: dict[str, ToolDef] = {
        "scan_section": ToolDef(
            spec=ToolSpec(
                name="scan_section",
                description="Scan a warehouse section to list items and their status.",
                parameters=SectionParams,
            ),
            callable=lambda **kw: db.scan_section(kw["section_id"]),
        ),
        "check_item": ToolDef(
            spec=ToolSpec(
                name="check_item",
                description=(
                    "Investigate a flagged item from a prior scan. "
                    "Only works for items that showed FLAG status."
                ),
                parameters=ItemParams,
            ),
            callable=lambda **kw: db.check_item(kw["item_id"]),
        ),
        "file_report": ToolDef(
            spec=ToolSpec(
                name="file_report",
                description="File the final audit report with all findings.",
                parameters=AuditReportParams,
            ),
            callable=lambda **kw: db.file_report(kw.get("findings", "")),
        ),
    }
    workflow = Workflow(
        name="inventory_audit",
        description="Scan 8 warehouse sections, investigate flagged items, file report",
        tools=tools,
        required_steps=["scan_section"],
        terminal_tool="file_report",
        system_prompt_template=(
            "You are a warehouse auditor. Scan all 8 sections (A through H) "
            "to identify discrepancies. For any item showing a FLAG status, "
            "investigate it with check_item to get details. After scanning all "
            "sections and investigating all flagged items, file a report "
            "summarizing all discrepancies found."
        ),
    )
    validate_state = lambda: (
        len(db.scanned) == 8
        and len(db.investigated) >= 4
    )
    return workflow, validate_state


def _validate_inventory_audit(args: dict[str, Any]) -> bool:
    text = (args.get("findings", "") or "").lower().replace(",", "")
    # Must mention at least 3 of the 5 flagged items by ID or description
    flagged_mentions = 0
    flag_indicators = [
        ("a-003", "lock nut"),    # shortage
        ("b-003", "o-ring"),      # shortage
        ("d-003", "power supply"),  # misplaced
        ("f-002", "goggles"),     # surplus
        ("h-004", "cut-off"),     # expired
    ]
    for item_id, item_desc in flag_indicators:
        if item_id in text or item_desc in text:
            flagged_mentions += 1
    return flagged_mentions >= 3


inventory_audit = EvalScenario(
    name="inventory_audit",
    description=(
        "Multi-turn warehouse audit — 8 section scans + flagged item investigation "
        "under tight budget. Tests cross-turn reasoning after compaction."
    ),
    workflow=_placeholder_workflow(
        "inventory_audit", "file_report", ["scan_section"],
    ),
    user_message=(
        "Perform a complete inventory audit of the warehouse. Scan all 8 "
        "sections (A through H), investigate every flagged discrepancy, "
        "and file a report with your findings."
    ),
    budget_tokens=750,
    max_iterations=20,
    validate=_validate_inventory_audit,
    build_workflow=_build_inventory_audit,
    tags=["stateful", "compaction"],
    ideal_iterations=13,
)


# ── Backend 13: SupplierResearchSystem (supplier_deep_dive) ───


# 5 suppliers with pricing, quality metrics, and customer references.
# Model must progressively narrow: all 5 quoted → top 3 by price quality-checked →
# top 2 by quality referenced → recommend best.
_SUPPLIER_QUOTES: dict[str, str] = {
    "globalchip": (
        "Quote — GlobalChip Semiconductor (Hsinchu)\n"
        "Component: FPGA-7200 | Unit Price: $18.40 (MOQ 1,000) | $16.90 (MOQ 5,000)\n"
        "Lead Time: 8 weeks | Payment: Net 30 | Shipping: FOB Hsinchu"
    ),
    "siliconedge": (
        "Quote — SiliconEdge Corp (San Jose)\n"
        "Component: FPGA-7200 | Unit Price: $21.50 (MOQ 500) | $19.80 (MOQ 3,000)\n"
        "Lead Time: 4 weeks | Payment: Net 45 | Shipping: DDP included"
    ),
    "nexwave": (
        "Quote — NexWave Electronics (Shenzhen)\n"
        "Component: FPGA-7200 | Unit Price: $15.20 (MOQ 2,000) | $14.50 (MOQ 10,000)\n"
        "Lead Time: 10 weeks | Payment: Net 30 | Shipping: FOB Shenzhen"
    ),
    "eurosilicon": (
        "Quote — EuroSilicon GmbH (Dresden)\n"
        "Component: FPGA-7200 | Unit Price: $22.80 (MOQ 500) | $20.10 (MOQ 5,000)\n"
        "Lead Time: 6 weeks | Payment: Net 60 | Shipping: DDP Europe"
    ),
    "quantumfab": (
        "Quote — QuantumFab Ltd (Penang)\n"
        "Component: FPGA-7200 | Unit Price: $16.80 (MOQ 1,000) | $15.60 (MOQ 5,000)\n"
        "Lead Time: 9 weeks | Payment: Net 30 | Shipping: FOB Penang"
    ),
}

_SUPPLIER_QUALITY: dict[str, str] = {
    "globalchip": (
        "Quality Report — GlobalChip Semiconductor\n"
        "Defect Rate: 0.3% | Yield: 97.2% | ISO 9001 + AS9100 certified\n"
        "Failure Mode: 68% ESD, 32% solder joint | MTBF: 120,000 hrs\n"
        "Audit Score: 94/100 (2024-11) | Corrective Actions: 0 open"
    ),
    "siliconedge": (
        "Quality Report — SiliconEdge Corp\n"
        "Defect Rate: 0.8% | Yield: 95.1% | ISO 9001 certified\n"
        "Failure Mode: 45% thermal, 55% bonding | MTBF: 85,000 hrs\n"
        "Audit Score: 87/100 (2024-09) | Corrective Actions: 2 open"
    ),
    "nexwave": (
        "Quality Report — NexWave Electronics\n"
        "Defect Rate: 0.4% | Yield: 96.8% | ISO 9001 + IATF 16949 certified\n"
        "Failure Mode: 71% contamination, 29% die crack | MTBF: 110,000 hrs\n"
        "Audit Score: 92/100 (2024-10) | Corrective Actions: 1 open"
    ),
    "eurosilicon": (
        "Quality Report — EuroSilicon GmbH\n"
        "Defect Rate: 0.2% | Yield: 98.1% | ISO 9001 + IATF 16949 + AS9100\n"
        "Failure Mode: 80% test escape, 20% marking | MTBF: 145,000 hrs\n"
        "Audit Score: 97/100 (2024-12) | Corrective Actions: 0 open"
    ),
    "quantumfab": (
        "Quality Report — QuantumFab Ltd\n"
        "Defect Rate: 0.6% | Yield: 96.0% | ISO 9001 certified\n"
        "Failure Mode: 50% wirebond, 50% passivation | MTBF: 95,000 hrs\n"
        "Audit Score: 89/100 (2024-08) | Corrective Actions: 1 open"
    ),
}

_SUPPLIER_REFERENCES: dict[str, str] = {
    "globalchip": (
        "References — GlobalChip Semiconductor\n"
        "1. Aerodyne Systems (defense): 3 years, 50K units/yr, 'excellent consistency'\n"
        "2. MedTech Instruments: 2 years, 20K units/yr, 'responsive to quality issues'\n"
        "3. AutoDrive Corp: 1 year, 80K units/yr, 'met all delivery windows'"
    ),
    "siliconedge": (
        "References — SiliconEdge Corp\n"
        "1. CloudScale Inc: 4 years, 30K units/yr, 'good domestic support'\n"
        "2. DroneWorks Ltd: 1 year, 10K units/yr, 'premium pricing but fast delivery'"
    ),
    "nexwave": (
        "References — NexWave Electronics\n"
        "1. TeleConnect (telecom): 5 years, 200K units/yr, 'best price-quality ratio'\n"
        "2. SmartGrid Energy: 3 years, 100K units/yr, 'reliable at high volume'\n"
        "3. RoboLogic Inc: 2 years, 40K units/yr, 'occasional lead time slip'"
    ),
    "eurosilicon": (
        "References — EuroSilicon GmbH\n"
        "1. AeroBus Industries (aerospace): 6 years, 25K units/yr, 'gold standard quality'\n"
        "2. SatelliteCom GmbH: 4 years, 15K units/yr, 'zero field failures'\n"
        "3. MediVision AG: 3 years, 10K units/yr, 'expensive but worth it for Class III'"
    ),
    "quantumfab": (
        "References — QuantumFab Ltd\n"
        "1. ConsumerTech Asia: 2 years, 500K units/yr, 'competitive for high volume'\n"
        "2. IoTLink Pte: 1 year, 60K units/yr, 'acceptable quality, good pricing'"
    ),
}

# Top 3 by price (at MOQ 5,000): NexWave $14.50, QuantumFab $15.60, GlobalChip $16.90
# Top 3 by quality (of those 3): GlobalChip 0.3%/94, NexWave 0.4%/92, QuantumFab 0.6%/89
# → Top 2 for references: GlobalChip, NexWave
# Best overall: NexWave (lowest price, strong quality, high-volume references)
# or GlobalChip (slightly higher price, best quality of top-3, aerospace refs)

_SUPPLIER_CANDIDATES = [
    "GlobalChip", "SiliconEdge", "NexWave", "EuroSilicon", "QuantumFab",
]


def _resolve_supplier_name(name: str) -> str | None:
    """Fuzzy-match supplier name to canonical key."""
    normalized = name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    for key in _SUPPLIER_QUOTES:
        if key in normalized or normalized in key:
            return key
    return None


class SupplierResearchSystem:
    """Stateful backend for supplier_deep_dive — progressive narrowing across 3 phases."""

    def __init__(self) -> None:
        self.quotes_fetched: list[str] = []
        self.quality_checked: list[str] = []
        self.references_checked: list[str] = []

    def list_candidates(self, component: str) -> str:
        return (
            f"Candidates for {component}:\n"
            + "\n".join(f"  {i+1}. {name}" for i, name in enumerate(_SUPPLIER_CANDIDATES))
        )

    def get_quote(self, supplier: str) -> str:
        key = _resolve_supplier_name(supplier)
        if key is None:
            return (
                f"Unknown supplier: '{supplier}'. "
                f"Available: {', '.join(_SUPPLIER_CANDIDATES)}."
            )
        if key not in self.quotes_fetched:
            self.quotes_fetched.append(key)
        return _SUPPLIER_QUOTES[key]

    def get_quality_report(self, supplier: str) -> str:
        key = _resolve_supplier_name(supplier)
        if key is None:
            return (
                f"Unknown supplier: '{supplier}'. "
                f"Available: {', '.join(_SUPPLIER_CANDIDATES)}."
            )
        if key not in self.quality_checked:
            self.quality_checked.append(key)
        return _SUPPLIER_QUALITY[key]

    def get_references(self, supplier: str) -> str:
        key = _resolve_supplier_name(supplier)
        if key is None:
            return (
                f"Unknown supplier: '{supplier}'. "
                f"Available: {', '.join(_SUPPLIER_CANDIDATES)}."
            )
        if key not in self.references_checked:
            self.references_checked.append(key)
        return _SUPPLIER_REFERENCES[key]

    def recommend(self, supplier: str, reasoning: str) -> str:
        return f"Recommendation recorded: {supplier}"


def _build_supplier_deep_dive() -> tuple[Workflow, callable]:
    db = SupplierResearchSystem()
    tools: dict[str, ToolDef] = {
        "list_candidates": ToolDef(
            spec=ToolSpec(
                name="list_candidates",
                description="List candidate suppliers for a component.",
                parameters=ComponentParams,
            ),
            callable=lambda **kw: db.list_candidates(kw["component"]),
        ),
        "get_quote": ToolDef(
            spec=ToolSpec(
                name="get_quote",
                description="Get pricing quote from a supplier.",
                parameters=SupplierNameParams,
            ),
            callable=lambda **kw: db.get_quote(kw["supplier"]),
        ),
        "get_quality_report": ToolDef(
            spec=ToolSpec(
                name="get_quality_report",
                description="Get quality metrics and audit data for a supplier.",
                parameters=SupplierNameParams,
            ),
            callable=lambda **kw: db.get_quality_report(kw["supplier"]),
        ),
        "get_references": ToolDef(
            spec=ToolSpec(
                name="get_references",
                description="Get customer references for a supplier.",
                parameters=SupplierNameParams,
            ),
            callable=lambda **kw: db.get_references(kw["supplier"]),
        ),
        "recommend": ToolDef(
            spec=ToolSpec(
                name="recommend",
                description=(
                    "Submit final supplier recommendation with justification "
                    "citing price, quality, and reference data."
                ),
                parameters=SupplierRecommendParams,
            ),
            callable=lambda **kw: db.recommend(
                kw.get("supplier", ""), kw.get("reasoning", ""),
            ),
        ),
    }
    workflow = Workflow(
        name="supplier_deep_dive",
        description=(
            "Evaluate 5 suppliers for FPGA-7200 across pricing, quality, and "
            "references, then recommend the best one"
        ),
        tools=tools,
        required_steps=["list_candidates", "get_quote"],
        terminal_tool="recommend",
        system_prompt_template=(
            "You are a procurement analyst evaluating suppliers for the "
            "FPGA-7200 component. Follow this process:\n"
            "1. List all candidates\n"
            "2. Get quotes from all 5 suppliers\n"
            "3. Get quality reports for the top 3 suppliers by price "
            "(lowest unit price at highest MOQ)\n"
            "4. Get references for the top 2 suppliers by quality "
            "(lowest defect rate among the 3 quality-checked)\n"
            "5. Recommend the best supplier, citing specific metrics from "
            "all three phases (price, quality, references)"
        ),
    )
    validate_state = lambda: (
        len(db.quotes_fetched) >= 5
        and len(db.quality_checked) >= 3
        and len(db.references_checked) >= 2
    )
    return workflow, validate_state


def _validate_supplier_deep_dive(args: dict[str, Any]) -> bool:
    text = f"{args.get('supplier', '')} {args.get('reasoning', '')}".lower()
    # Best choices are NexWave or GlobalChip (both are defensible)
    has_supplier = any(s in text for s in ["nexwave", "nex wave", "globalchip", "global chip"])
    # Must cite at least one metric from pricing AND quality
    has_price = any(t in text for t in [
        "14.50", "15.20", "16.90", "18.40",
    ])
    has_quality = any(t in text for t in [
        "0.3%", "0.4%", "defect", "yield", "mtbf", "audit score",
        "as9100", "iatf",
    ])
    return has_supplier and has_price and has_quality


supplier_deep_dive = EvalScenario(
    name="supplier_deep_dive",
    description=(
        "Multi-phase supplier evaluation with progressive narrowing — "
        "pricing, quality, references under tight budget. Tests cross-turn "
        "reasoning after compaction."
    ),
    workflow=_placeholder_workflow(
        "supplier_deep_dive", "recommend", ["list_candidates", "get_quote"],
    ),
    user_message=(
        "We need to source 5,000 FPGA-7200 components. List all candidate "
        "suppliers, get quotes from all 5, then narrow to the top 3 by price "
        "for quality checks, then the top 2 by quality for references. "
        "Recommend the best supplier with justification."
    ),
    budget_tokens=800,
    max_iterations=20,
    validate=_validate_supplier_deep_dive,
    build_workflow=_build_supplier_deep_dive,
    tags=["stateful", "compaction", "reasoning"],
    ideal_iterations=12,
)


# ── Backend 9: TravelBookingSystem ─────────────────────────────


class TravelBookingSystem:
    def __init__(self) -> None:
        self.forecasts_fetched: list[str] = []
        self.flights_booked: list[tuple[str, str]] = []
        self.hotels_checked: list[str] = []
        self.conversions: list[tuple[str, str, str]] = []

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


def _build_relevance_detection_stateful() -> tuple[Workflow, callable]:
    db = TravelBookingSystem()
    tools: dict[str, ToolDef] = {
        "get_forecast": ToolDef(
            spec=ToolSpec(
                name="get_forecast",
                description="Get weather forecast for a city.",
                parameters=CityParams,
            ),
            callable=lambda **kw: db.get_forecast(kw["city"]),
        ),
        "book_flight": ToolDef(
            spec=ToolSpec(
                name="book_flight",
                description="Book a flight between two cities.",
                parameters=FlightParams,
            ),
            callable=lambda **kw: db.book_flight(kw["origin"], kw["destination"]),
        ),
        "check_hotel": ToolDef(
            spec=ToolSpec(
                name="check_hotel",
                description="Check hotel availability in a city.",
                parameters=HotelParams,
            ),
            callable=lambda **kw: db.check_hotel(kw["city"], kw["checkin"]),
        ),
        "convert_currency": ToolDef(
            spec=ToolSpec(
                name="convert_currency",
                description="Convert an amount between currencies.",
                parameters=CurrencyParams,
            ),
            callable=lambda **kw: db.convert_currency(
                kw["amount"], kw["from_currency"], kw["to_currency"],
            ),
        ),
        "decline": ToolDef(
            spec=ToolSpec(
                name="decline",
                description="Call this when none of the available tools are relevant to the user's request.",
                parameters=ReasonParams,
            ),
            callable=lambda **kw: db.decline(kw.get("reason", "")),
        ),
    }
    workflow = Workflow(
        name="relevance_detection_stateful",
        description="Travel tools that are irrelevant to the user's question",
        tools=tools,
        required_steps=[],
        terminal_tool="decline",
        system_prompt_template=(
            "You are a helpful assistant. You have access to travel-related "
            "tools. If the user's request cannot be answered using the "
            "available tools, call the decline tool to explain why. "
            "Do NOT call a tool unless it is directly relevant."
        ),
    )
    validate_state = lambda: (
        db.forecasts_fetched == []
        and db.flights_booked == []
        and db.hotels_checked == []
        and db.conversions == []
    )
    return workflow, validate_state


relevance_detection_stateful = EvalScenario(
    name="relevance_detection_stateful",
    description="Stateful relevance detection — verifies no travel tools were called.",
    workflow=_placeholder_workflow("relevance_detection_stateful", "decline"),
    user_message="What is the square root of 144?",
    max_iterations=5,
    validate=lambda args: bool(args.get("reason", "").strip()),
    build_workflow=_build_relevance_detection_stateful,
    tags=["stateful", "model_quality"],
    ideal_iterations=1,
)
