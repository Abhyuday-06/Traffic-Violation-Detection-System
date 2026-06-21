from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "violations.db"
EVIDENCE_ROOT = PROJECT_ROOT / "evidence"
IMAGE_DIR = EVIDENCE_ROOT / "images"
PDF_DIR = EVIDENCE_ROOT / "pdfs"

UNKNOWN_LICENSE_PLATE = "UNKNOWN_REVIEW_REQUIRED"
PDF_HEADER = "TRAFFIC LAW ENFORCEMENT E-CHALLAN"

# Default fine amounts (INR) per violation type – adjust as required.
PENALTY_TABLE: dict[str, int] = {
    "Helmet Violation":                500,
    "Rider: No Helmet":                500,
    "Pillion: No Helmet":              500,
    "Triple Riding Violation":        1000,
    "Seatbelt Violation":              500,
    "Wrong-side Violation":           1000,
    "Red-light / Stop-line Violation": 500,
    "Illegal Parking Violation":       500,
}
DEFAULT_PENALTY = 250   # fallback for unknown violation types


def _ensure_directories() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> Path:
    """
    Initialize the local SQLite challan database and return its path.
    """
    database_path = Path(db_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS challans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                license_plate TEXT NOT NULL,
                crop_path TEXT NOT NULL,
                pdf_path TEXT NOT NULL
            )
            """
        )
        connection.commit()

    return database_path


def _validate_frame_crop(frame_crop: np.ndarray) -> None:
    if frame_crop is None:
        raise ValueError("frame_crop must not be None")
    if not isinstance(frame_crop, np.ndarray):
        raise TypeError("frame_crop must be a numpy.ndarray")
    if frame_crop.ndim not in {2, 3}:
        raise ValueError("frame_crop must be a grayscale or BGR image")
    if frame_crop.size == 0:
        raise ValueError("frame_crop must not be empty")


def _sanitize_license_plate(license_plate: str | None) -> str:
    if not license_plate:
        return UNKNOWN_LICENSE_PLATE

    cleaned = "".join(character for character in license_plate.upper() if character.isalnum() or character == "_")
    return cleaned or UNKNOWN_LICENSE_PLATE


def _insert_challan_stub(
    db_path: Path,
    timestamp: str,
    violation_type: str,
    confidence: float,
    license_plate: str,
) -> int:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO challans (
                timestamp,
                violation_type,
                confidence,
                license_plate,
                crop_path,
                pdf_path
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (timestamp, violation_type, float(confidence), license_plate, "", ""),
        )
        connection.commit()
        return int(cursor.lastrowid)


def _update_challan_paths(db_path: Path, challan_id: int, crop_path: Path, pdf_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE challans
            SET crop_path = ?, pdf_path = ?
            WHERE id = ?
            """,
            (str(crop_path), str(pdf_path), int(challan_id)),
        )
        connection.commit()


def _save_crop_image(frame_crop: np.ndarray, challan_id: int, timestamp: str) -> Path:
    image_path = IMAGE_DIR / f"challan_{challan_id:06d}_{timestamp}.jpg"
    success = cv2.imwrite(str(image_path), frame_crop)
    if not success:
        raise RuntimeError(f"Failed to save crop image: {image_path}")
    return image_path


def _load_reportlab() -> dict[str, Any]:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "reportlab is required to generate challan PDFs. Install it with: pip install reportlab"
        ) from exc

    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "A4": A4,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "inch": inch,
        "Image": Image,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
    }


def _make_styles(reportlab: dict[str, Any]) -> dict[str, Any]:
    sample_styles = reportlab["getSampleStyleSheet"]()
    paragraph_style = reportlab["ParagraphStyle"]
    return {
        "header": paragraph_style(
            "ChallanHeader",
            parent=sample_styles["Title"],
            alignment=reportlab["TA_CENTER"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=18,
        ),
        "body": paragraph_style(
            "ChallanBody",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
        ),
        "label": paragraph_style(
            "ChallanLabel",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
        ),
        "table_header": paragraph_style(
            "ChallanTableHeader",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
        ),
        "table_body": paragraph_style(
            "ChallanTableBody",
            parent=sample_styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
        ),
        "total": paragraph_style(
            "ChallanTotal",
            parent=sample_styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
        ),
    }


def _scaled_image_flowable(image_path: Path, max_width: float, max_height: float, image_class: Any) -> Any:
    crop = cv2.imread(str(image_path))
    if crop is None:
        raise RuntimeError(f"Unable to read saved crop image: {image_path}")

    height, width = crop.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    return image_class(str(image_path), width=width * scale, height=height * scale)


def generate_challan_pdf(
    challan_id: int,
    timestamp: str,
    violations: list[dict[str, Any]],
    license_plate: str,
    crop_path: str | Path,
) -> Path:
    """
    Generate a formal consolidated e-Challan PDF with an itemized violations table.

    Parameters
    ----------
    violations : list[dict]
        Each entry must have at least ``type`` (str) and ``confidence`` (float).
        Optional key ``penalty`` (int, INR) is used if present; otherwise the
        global PENALTY_TABLE is consulted.
    """
    _ensure_directories()

    reportlab = _load_reportlab()
    colors = reportlab["colors"]
    inch = reportlab["inch"]
    image_class = reportlab["Image"]
    paragraph = reportlab["Paragraph"]
    simple_doc_template = reportlab["SimpleDocTemplate"]
    spacer = reportlab["Spacer"]
    table = reportlab["Table"]
    table_style_cls = reportlab["TableStyle"]

    crop_path = Path(crop_path)
    safe_timestamp = "".join(character if character.isalnum() else "_" for character in timestamp)
    pdf_path = PDF_DIR / f"challan_{challan_id:06d}_{safe_timestamp}.pdf"
    styles = _make_styles(reportlab)

    document = simple_doc_template(
        str(pdf_path),
        pagesize=reportlab["A4"],
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title=f"E-Challan {challan_id:06d}",
        author="Automated Traffic Violation Detection System",
    )

    # ── summary header table ─────────────────────────────────────────────────
    summary_rows = [
        [paragraph("Challan ID",       styles["label"]), paragraph(f"{challan_id:06d}",  styles["body"])],
        [paragraph("Date/Time",        styles["label"]), paragraph(timestamp,             styles["body"])],
        [paragraph("Number Plate",     styles["label"]), paragraph(license_plate,         styles["body"])],
        [paragraph("Total Offences",   styles["label"]), paragraph(str(len(violations)),  styles["body"])],
    ]
    summary_table = table(summary_rows, colWidths=[2.0 * inch, 4.2 * inch], hAlign="CENTER")
    summary_table.setStyle(
        table_style_cls(
            [
                ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#7A7A7A")),
                ("BACKGROUND",  (0, 0), (0, -1),  colors.HexColor("#EDEDED")),
                ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING",(0, 0), (-1, -1), 8),
                ("TOPPADDING",  (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING",(0, 0),(-1, -1), 7),
            ]
        )
    )

    # ── itemized violations table ────────────────────────────────────────────
    viol_header = [
        paragraph("#",                styles["table_header"]),
        paragraph("Violation",        styles["table_header"]),
        paragraph("Confidence",       styles["table_header"]),
        paragraph("Penalty (INR)",    styles["table_header"]),
    ]
    viol_rows: list[list[Any]] = [viol_header]
    total_penalty = 0

    for idx, v in enumerate(violations, start=1):
        vtype      = str(v.get("type", "Unknown Violation"))
        conf       = float(v.get("confidence") or 0.0)
        penalty    = int(v.get("penalty", PENALTY_TABLE.get(vtype, DEFAULT_PENALTY)))
        total_penalty += penalty
        viol_rows.append([
            paragraph(str(idx),              styles["table_body"]),
            paragraph(vtype,                 styles["table_body"]),
            paragraph(f"{conf * 100:.1f}%",  styles["table_body"]),
            paragraph(f"Rs. {penalty:,}",    styles["table_body"]),
        ])

    # Total row
    viol_rows.append([
        paragraph("",                         styles["total"]),
        paragraph("TOTAL FINE PAYABLE",       styles["total"]),
        paragraph("",                         styles["total"]),
        paragraph(f"Rs. {total_penalty:,}",   styles["total"]),
    ])

    total_row_idx = len(viol_rows) - 1
    viol_table = table(
        viol_rows,
        colWidths=[0.35 * inch, 3.2 * inch, 1.2 * inch, 1.45 * inch],
        hAlign="CENTER",
    )
    viol_table.setStyle(
        table_style_cls(
            [
                # header row
                ("BACKGROUND",   (0, 0), (-1, 0),            colors.HexColor("#2C2C2C")),
                ("TEXTCOLOR",    (0, 0), (-1, 0),            colors.white),
                # body rows – alternating shade
                ("ROWBACKGROUNDS", (0, 1), (-1, total_row_idx - 1),
                 [colors.HexColor("#FFFFFF"), colors.HexColor("#F5F5F5")]),
                # total row
                ("BACKGROUND",   (0, total_row_idx), (-1, total_row_idx), colors.HexColor("#FFE0E0")),
                ("TEXTCOLOR",    (0, total_row_idx), (-1, total_row_idx), colors.HexColor("#CC0000")),
                # grid and padding
                ("GRID",         (0, 0), (-1, -1),           0.5, colors.HexColor("#AAAAAA")),
                ("LINEABOVE",    (0, total_row_idx), (-1, total_row_idx), 1.5, colors.HexColor("#CC0000")),
                ("VALIGN",       (0, 0), (-1, -1),           "MIDDLE"),
                ("LEFTPADDING",  (0, 0), (-1, -1),           6),
                ("RIGHTPADDING", (0, 0), (-1, -1),           6),
                ("TOPPADDING",   (0, 0), (-1, -1),           5),
                ("BOTTOMPADDING",(0, 0), (-1, -1),           5),
            ]
        )
    )

    # ── evidence image ───────────────────────────────────────────────────────
    evidence_image = _scaled_image_flowable(
        crop_path,
        max_width=6.2 * inch,
        max_height=3.4 * inch,
        image_class=image_class,
    )
    evidence_table = table([[evidence_image]], hAlign="CENTER")
    evidence_table.setStyle(
        table_style_cls(
            [
                ("BOX",          (0, 0), (-1, -1), 0.75, colors.HexColor("#555555")),
                ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
                ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ]
        )
    )

    story = [
        paragraph(PDF_HEADER, styles["header"]),
        summary_table,
        spacer(1, 0.25 * inch),
        paragraph("Itemized Violations", styles["label"]),
        spacer(1, 0.08 * inch),
        viol_table,
        spacer(1, 0.28 * inch),
        paragraph("Visual Evidence (Best Frame Crop)", styles["label"]),
        spacer(1, 0.10 * inch),
        evidence_table,
    ]

    document.build(story)
    return pdf_path


def save_violation_record(
    violations: list[dict[str, Any]] | str,
    confidence: float = 0.0,
    license_plate: str | None = None,
    frame_crop: np.ndarray | None = None,
    db_path: str | Path = DEFAULT_DB_PATH,
    # Legacy keyword kept for backward compatibility
    violation_type: str | None = None,
) -> dict[str, Any]:
    """
    Persist a consolidated e-Challan: save the best crop image, generate a
    multi-violation itemized PDF, and write the record to SQLite.

    Parameters
    ----------
    violations : list[dict] | str
        Preferred form: a list of violation dicts, each with at minimum a
        ``"type"`` key and optionally ``"confidence"`` and ``"penalty"``.
        For backward compatibility, a bare string violation type is also
        accepted (it is wrapped into a single-item list automatically).
    confidence : float
        Used only when *violations* is a bare string (legacy mode).
    license_plate : str | None
    frame_crop : np.ndarray
        Best captured frame crop for this incident.
    db_path : str | Path
    violation_type : str | None
        Deprecated – ignored when *violations* is a list.
    """
    # ── normalise the violations argument ────────────────────────────────────
    if isinstance(violations, str):
        # Legacy single-string call – wrap into list
        violations = [{"type": violations, "confidence": confidence}]
    elif not violations:
        violations = [{"type": violation_type or "Unknown Violation", "confidence": confidence}]

    if frame_crop is None:
        raise ValueError("frame_crop must not be None")
    _validate_frame_crop(frame_crop)
    _load_reportlab()
    _ensure_directories()

    # Use the worst (highest-penalty) violation as the DB summary string
    primary = max(
        violations,
        key=lambda v: PENALTY_TABLE.get(str(v.get("type", "")), DEFAULT_PENALTY),
    )
    primary_type = str(primary.get("type", "Unknown Violation"))
    primary_conf = float(primary.get("confidence") or confidence or 0.0)
    all_types_str = " | ".join(str(v.get("type", "")) for v in violations)

    database_path = initialize_database(db_path)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
    display_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    normalized_plate = _sanitize_license_plate(license_plate)

    challan_id = _insert_challan_stub(
        db_path=database_path,
        timestamp=display_timestamp,
        violation_type=all_types_str,
        confidence=primary_conf,
        license_plate=normalized_plate,
    )

    crop_path = _save_crop_image(frame_crop, challan_id, timestamp)
    pdf_path = generate_challan_pdf(
        challan_id=challan_id,
        timestamp=display_timestamp,
        violations=violations,
        license_plate=normalized_plate,
        crop_path=crop_path,
    )
    _update_challan_paths(database_path, challan_id, crop_path, pdf_path)

    return {
        "id": challan_id,
        "timestamp": display_timestamp,
        "violation_type": all_types_str,
        "confidence": primary_conf,
        "license_plate": normalized_plate,
        "crop_path": str(crop_path),
        "pdf_path": str(pdf_path),
        "db_path": str(database_path),
    }


initialize_database()
