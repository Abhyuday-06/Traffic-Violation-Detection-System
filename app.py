from __future__ import annotations

import html
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as sic
    _SIC_AVAILABLE = True
except ImportError:
    _SIC_AVAILABLE = False

from utils.evidence_generator import DEFAULT_DB_PATH, initialize_database, save_violation_record


PROJECT_ROOT = Path(__file__).resolve().parent
UNKNOWN_PLATE_TEXT = "MANUAL_REVIEW"
IMAGE_UPLOAD_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_UPLOAD_TYPES = ["mp4", "avi"]


def configure_page() -> None:
    st.set_page_config(
        page_title="Traffic Enforcement System",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .block-container { padding-top: 1.2rem; padding-bottom: 1.5rem; }
            .activity-feed {
                max-height: 560px;
                overflow-y: auto;
                border: 1px solid rgba(49, 51, 63, 0.18);
                border-radius: 8px;
                padding: 0.75rem;
                background: rgba(250, 250, 250, 0.72);
            }
            .activity-alert {
                border-bottom: 1px solid rgba(49, 51, 63, 0.12);
                padding: 0.55rem 0;
                font-size: 0.92rem;
                line-height: 1.35;
            }
            .activity-alert:last-child { border-bottom: 0; }
            .muted { color: rgba(49, 51, 63, 0.68); font-size: 0.86rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> dict[str, Any]:
    st.sidebar.title("Control Panel")
    operation_mode = st.sidebar.selectbox(
        "Operation Mode",
        ["Evidence Image or Video Analysis", "System Performance Analytics Dashboard"],
    )

    st.sidebar.subheader("Model Paths")
    primary_model_path = st.sidebar.text_input("Primary Tracker", value="yolov8s.pt")
    helmet_model_path = st.sidebar.text_input("Helmet Weights", value=str(PROJECT_ROOT / "models" / "helmet_best.pt"))
    seatbelt_model_path = st.sidebar.text_input("Seatbelt Weights", value=str(PROJECT_ROOT / "models" / "seatbelt_best.pt"))
    plate_model_path = st.sidebar.text_input("Plate Weights", value=str(PROJECT_ROOT / "models" / "lp_best.pt"))

    st.sidebar.subheader("Confidence Thresholds")
    primary_confidence = st.sidebar.slider("Primary Detection", 0.10, 0.95, 0.35, 0.05)
    helmet_confidence = st.sidebar.slider("Helmet Classifier", 0.10, 0.95, 0.55, 0.05)
    seatbelt_confidence = st.sidebar.slider("Seatbelt Classifier", 0.10, 0.95, 0.50, 0.05)
    plate_confidence = st.sidebar.slider("Plate Localizer", 0.10, 0.95, 0.25, 0.05)
    ocr_confidence = st.sidebar.slider("OCR Acceptance", 0.10, 0.95, 0.40, 0.05)

    st.sidebar.subheader("Runtime")
    device_choice = st.sidebar.selectbox("Inference Device", ["auto", "cuda", "cpu"])
    manual_traffic_light = st.sidebar.selectbox(
        "Manual Traffic Light State",
        ["Auto-Detect", "RED", "YELLOW", "GREEN"],
        help="Override the automatic traffic light detection logic. Use if the traffic light is outside the camera view."
    )
    frame_stride = st.sidebar.slider("Frame Stride", 1, 10, 1)
    max_frames = st.sidebar.number_input("Max Frames", min_value=0, max_value=500000, value=0, step=100)
    generate_evidence = st.sidebar.checkbox("Generate OCR and PDF Evidence", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🛠 Developer Tools")
    dev_mode = st.sidebar.toggle(
        "[DEV MODE] Enable Pipeline Debugger",
        value=False,
        help=(
            "When active, displays raw track detections, intermediate image crops "
            "passed to each sub-model, OCR preprocessing variants, and live spatial "
            "heuristic states in an expandable panel below the video player."
        ),
    )

    return {
        "operation_mode": operation_mode,
        "primary_model_path": primary_model_path,
        "helmet_model_path": helmet_model_path,
        "seatbelt_model_path": seatbelt_model_path,
        "plate_model_path": plate_model_path,
        "primary_confidence": primary_confidence,
        "helmet_confidence": helmet_confidence,
        "seatbelt_confidence": seatbelt_confidence,
        "plate_confidence": plate_confidence,
        "ocr_confidence": ocr_confidence,
        "device": None if device_choice == "auto" else device_choice,
        "manual_traffic_light": manual_traffic_light,
        "frame_stride": frame_stride,
        "max_frames": int(max_frames),
        "generate_evidence": generate_evidence,
        "dev_mode": dev_mode,
    }


@st.cache_resource(show_spinner=False)
def load_enforcement_engine(
    primary_model_path: str,
    helmet_model_path: str,
    seatbelt_model_path: str,
    primary_confidence: float,
    helmet_confidence: float,
    seatbelt_confidence: float,
    device: str | None,
) -> Any:
    from core.detection_engine import TrafficEnforcementEngine

    return TrafficEnforcementEngine(
        primary_model_path=primary_model_path,
        helmet_model_path=helmet_model_path,
        seatbelt_model_path=seatbelt_model_path,
        primary_conf_threshold=primary_confidence,
        helmet_conf_threshold=helmet_confidence,
        seatbelt_conf_threshold=seatbelt_confidence,
        device=device,
    )


def reset_engine_runtime_state(engine: Any) -> None:
    engine.frame_index = 0
    engine.centroid_history.clear()
    engine.parking_state.clear()


def persist_uploaded_video(uploaded_file: Any) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def is_image_upload(uploaded_file: Any) -> bool:
    suffix = Path(uploaded_file.name).suffix.lower().lstrip(".")
    return suffix in IMAGE_UPLOAD_TYPES


def decode_uploaded_image(uploaded_file: Any) -> np.ndarray | None:
    image_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)


def extract_first_frame(uploaded_file: Any) -> np.ndarray | None:
    """
    Write the uploaded video to a temp file and read its first frame with OpenCV.
    Returns a BGR numpy array or None if reading fails.
    """
    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        ok, frame = cap.read()
        cap.release()
        Path(tmp_path).unlink(missing_ok=True)
        return frame if ok else None
    except Exception:
        return None


def auto_calibrate_stop_line(
    frame: np.ndarray,
) -> list[tuple[float, float]] | None:
    """
    Run Canny edge detection + probabilistic Hough lines to estimate the longest
    near-horizontal line in the frame and return it as a thin 4-point polygon
    that can be passed to the engine as a stop-line override.

    Returns a list of 4 (x, y) pixel tuples or None if no line is found.
    """
    h, w = frame.shape[:2]
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (5, 5), 0)
    edges  = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(w * 0.25),
        maxLineGap=30,
    )
    if lines is None:
        return None

    # Keep only near-horizontal lines (angle within ±20° of horizontal)
    best: tuple[int, int, int, int] | None = None
    best_len = 0.0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if angle > 20:
            continue
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length > best_len:
            best_len = length
            best = (x1, y1, x2, y2)

    if best is None:
        return None

    x1, y1, x2, y2 = best
    thickness = max(6, int(h * 0.012))   # ~1.2% of frame height
    return [
        (float(min(x1, x2)), float(min(y1, y2) - thickness)),
        (float(max(x1, x2)), float(min(y1, y2) - thickness)),
        (float(max(x1, x2)), float(max(y1, y2) + thickness)),
        (float(min(x1, x2)), float(max(y1, y2) + thickness)),
    ]


def render_roi_calibration_ui(reference_frame: np.ndarray) -> None:
    """
    Render the 'Define Spatial Zones (ROI)' expander.
    """
    h, w = reference_frame.shape[:2]

    with st.expander("📐 Define Spatial Zones (ROI) — Human-in-the-Loop Calibration", expanded=True):
        st.markdown(
            """
            Click **4 points** on the image to define each zone polygon.
            Select which point to place using the radio buttons below. You can adjust previously placed points at any time.
            Click **Reset to Auto-Calibration** to let OpenCV estimate the stop-line.
            """
        )

        # --- Zone selector ------------------------------------------------
        zone = st.radio(
            "Active zone being defined:",
            ["Stop Line", "No-Parking Zone"],
            horizontal=True,
            key="roi_zone_radio",
        )
        zone_key = "custom_roi_stop" if zone == "Stop Line" else "custom_roi_parking"

        current_pts = list(st.session_state.get(zone_key, []))
        # Ensure we always have exactly 4 slots (could be None)
        while len(current_pts) < 4:
            current_pts.append(None)
        # If it somehow got longer, truncate
        current_pts = current_pts[:4]

        idx_key = f"active_point_idx_{zone_key}"
        if idx_key not in st.session_state:
            st.session_state[idx_key] = 0

        # Point selector (uses a dynamic key to allow programmatic updates from the click handler)
        dynamic_radio_key = f"radio_widget_{zone_key}_{st.session_state[idx_key]}"
        active_point_idx = st.radio(
            f"Select point to place for {zone}:",
            options=[0, 1, 2, 3],
            index=st.session_state[idx_key],
            format_func=lambda i: f"Point {i+1}" + (" (Set)" if current_pts[i] is not None else " (Empty)"),
            horizontal=True,
            key=dynamic_radio_key,
        )
        # Keep the session state synced with the user's manual radio selection
        st.session_state[idx_key] = active_point_idx

        col_reset, col_clear = st.columns([1, 1])
        with col_reset:
            if st.button("🔄 Reset to Auto-Calibration (Stop Line)", use_container_width=True):
                auto = auto_calibrate_stop_line(reference_frame)
                if auto:
                    st.session_state["custom_roi_stop"] = auto
                    st.success("Auto-calibration applied to Stop Line.")
                    st.rerun()
                else:
                    st.warning("No dominant horizontal line found – try adjusting the video or define manually.")
        with col_clear:
            if st.button(f"✖ Clear {zone}", use_container_width=True):
                st.session_state[zone_key] = [None, None, None, None]
                st.session_state[idx_key] = 0
                st.session_state.pop(f"last_click_{zone_key}", None)
                st.rerun()

        # --- Build the preview frame with existing polygons drawn ----------
        preview = reference_frame.copy()
        
        stop_pts = st.session_state.get("custom_roi_stop", [])
        park_pts = st.session_state.get("custom_roi_parking", [])
        
        valid_stop_pts = [pt for pt in stop_pts if pt is not None]
        valid_park_pts = [pt for pt in park_pts if pt is not None]

        if len(valid_stop_pts) >= 2:
            pts = np.array(valid_stop_pts, dtype=np.int32)
            cv2.polylines(preview, [pts], len(valid_stop_pts) == 4, (0, 255, 255), 3)
        if len(valid_park_pts) >= 2:
            pts = np.array(valid_park_pts, dtype=np.int32)
            cv2.polylines(preview, [pts], len(valid_park_pts) == 4, (128, 0, 255), 3)

        # Draw individual click dots + ordinal labels
        for label_pts_raw, color in [
            (stop_pts, (0, 255, 255)),
            (park_pts, (128, 0, 255)),
        ]:
            for i, pt in enumerate(label_pts_raw):
                if pt is None:
                    continue
                px_, py_ = pt
                cv2.circle(preview, (int(px_), int(py_)), 7, color, -1)
                cv2.putText(preview, str(i + 1), (int(px_) + 9, int(py_) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        # Status badges
        stop_status  = f"✅ {len(valid_stop_pts)}/4 pts" if len(valid_stop_pts) == 4 else f"⏳ {len(valid_stop_pts)}/4 pts"
        park_status  = f"✅ {len(valid_park_pts)}/4 pts" if len(valid_park_pts) == 4 else f"⏳ {len(valid_park_pts)}/4 pts"
        st.caption(f"🟦 Stop Line: {stop_status}   🟪 No-Parking: {park_status}")

        # Convert to RGB for proper color rendering in Streamlit's web components
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        # Scale the image down to exactly 800px width. sic component sometimes hallucinates coordinates
        # when we rely on its internal `use_column_width=True` CSS scaling. By doing strict Numpy resizing, 
        # we mathematically lock the coordinates.
        display_width = 800
        scale_ratio = display_width / w
        display_height = int(h * scale_ratio)
        preview_resized = cv2.resize(preview_rgb, (display_width, display_height))

        # --- Interactive image component -----------------------------------
        if _SIC_AVAILABLE:
            # Render exactly at natural size of preview_resized
            click = sic(preview_resized, key=f"roi_click_{zone_key}")
            if click is not None:
                last_click_key = f"last_click_{zone_key}"
                last_click = st.session_state.get(last_click_key)
                if last_click != click:
                    # It's a new click
                    st.session_state[last_click_key] = click
                    # Scale coordinates strictly back up to the original video frame size
                    cx = float(click["x"]) / scale_ratio
                    cy = float(click["y"]) / scale_ratio
                    
                    # Update the specific point
                    current_pts[active_point_idx] = (cx, cy)
                    st.session_state[zone_key] = current_pts
                    
                    # Auto-advance to the next empty point, or next point
                    next_idx = active_point_idx
                    if active_point_idx < 3:
                        next_idx = active_point_idx + 1
                    
                    # Optional: find first empty point to auto-advance to
                    for offset in range(1, 4):
                        check_idx = (active_point_idx + offset) % 4
                        if current_pts[check_idx] is None:
                            next_idx = check_idx
                            break
                            
                    st.session_state[idx_key] = next_idx
                    st.rerun()
        else:
            st.image(preview, channels="BGR", use_container_width=True)
            st.warning(
                "`streamlit-image-coordinates` is not installed. "
                "Run `pip install streamlit-image-coordinates` and restart the app."
            )


def crop_bbox(frame: np.ndarray, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def read_plate_text(vehicle_crop: np.ndarray, plate_model_path: str, plate_confidence: float, ocr_confidence: float) -> str:
    try:
        from core.ocr_engine import read_vehicle_license_plate

        result = read_vehicle_license_plate(
            vehicle_crop=vehicle_crop,
            model_path=plate_model_path,
            plate_confidence_threshold=plate_confidence,
            ocr_confidence_threshold=ocr_confidence,
        )
        return result.text
    except Exception:
        return UNKNOWN_PLATE_TEXT


def register_violation_evidence(
    violation: dict[str, Any],
    frame: np.ndarray,
    config: dict[str, Any],
) -> str:
    track_id = violation.get("track_id", "NA")
    violation_type = str(violation.get("type", "Unknown Violation"))
    confidence = float(violation.get("confidence") or 0.0)
    metadata = violation.get("metadata") or {}
    crop = crop_bbox(frame, violation.get("bbox", []))

    if crop is None:
        return f"ID {track_id}: {violation_type} - Evidence crop unavailable"

    ocr_bbox = metadata.get("vehicle_bbox") or violation.get("bbox", [])
    ocr_crop = crop_bbox(frame, ocr_bbox)
    if ocr_crop is None:
        ocr_crop = crop
    plate_text = read_plate_text(
        vehicle_crop=ocr_crop,
        plate_model_path=config["plate_model_path"],
        plate_confidence=config["plate_confidence"],
        ocr_confidence=config["ocr_confidence"],
    )

    if not config["generate_evidence"]:
        return f"ID {track_id}: {violation_type} - Plate: {plate_text}"

    try:
        record = save_violation_record(
            violation_type=violation_type,
            confidence=confidence,
            license_plate=plate_text,
            frame_crop=crop,
        )
        return f"ID {track_id}: {violation_type} - Plate: {plate_text} - PDF Generated #{record['id']:06d}"
    except Exception as exc:
        return f"ID {track_id}: {violation_type} - Plate: {plate_text} - Evidence pending: {exc}"


def render_activity_feed(alerts: list[str]) -> None:
    if not alerts:
        st.markdown('<div class="activity-feed"><span class="muted">No active violation alerts.</span></div>', unsafe_allow_html=True)
        return

    alert_items = []
    for alert in reversed(alerts[-60:]):
        alert_items.append(f'<div class="activity-alert">{html.escape(alert)}</div>')
    st.markdown(f'<div class="activity-feed">{"".join(alert_items)}</div>', unsafe_allow_html=True)


def render_debug_expander(debug_payload: dict[str, Any], frame_number: int | None = None) -> None:
    """Render the Developer Debug Data expander below the video player."""
    if not debug_payload:
        return

    title = "🔬 Developer Debug Data"
    if frame_number is not None:
        title += f" — Frame {frame_number}"

    with st.expander(title, expanded=True):
        # ---- Raw track detections ----------------------------------------
        raw_tracks = debug_payload.get("raw_tracks", [])
        if raw_tracks:
            st.markdown("**📡 Raw Track Detections**")
            st.dataframe(
                pd.DataFrame(raw_tracks)[["track_id", "class_name", "confidence", "bbox", "centroid"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No detections on this frame.")

        # ---- Tracking / spatial heuristic state --------------------------
        tracking_state = debug_payload.get("tracking_state", [])
        if tracking_state:
            st.markdown("**🗺 Tracking & Spatial Heuristics**")
            st.dataframe(
                pd.DataFrame(tracking_state),
                use_container_width=True,
                hide_index=True,
            )

        # ---- Helmet crops ------------------------------------------------
        helmet_crops = debug_payload.get("helmet_crops", [])
        if helmet_crops:
            st.markdown("**🪖 Helmet Model Input Crops**")
            cols = st.columns(min(len(helmet_crops), 4))
            for idx, entry in enumerate(helmet_crops):
                crop: np.ndarray | None = entry.get("crop")
                if crop is None or crop.size == 0:
                    continue
                col = cols[idx % len(cols)]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else crop
                conf_val = entry.get("confidence")
                caption = (
                    f"Rider ID:{entry['rider_track_id']} | Moto ID:{entry['motorcycle_track_id']}\n"
                    f"No-Helmet conf: {conf_val:.3f}" if conf_val is not None
                    else f"Rider ID:{entry['rider_track_id']} | No violation detected"
                )
                col.image(rgb, caption=caption, use_container_width=True)

        # ---- Seatbelt crops ----------------------------------------------
        seatbelt_crops = debug_payload.get("seatbelt_crops", [])
        if seatbelt_crops:
            st.markdown("**🪢 Seatbelt Model Input Crops**")
            cols = st.columns(min(len(seatbelt_crops), 4))
            for idx, entry in enumerate(seatbelt_crops):
                crop = entry.get("crop")
                if crop is None or crop.size == 0:
                    continue
                col = cols[idx % len(cols)]
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.ndim == 3 else crop
                conf_val = entry.get("confidence")
                caption = (
                    f"Car ID:{entry['car_track_id']}\nNo-Seatbelt conf: {conf_val:.3f}" if conf_val is not None
                    else f"Car ID:{entry['car_track_id']} | No violation detected"
                )
                col.image(rgb, caption=caption, use_container_width=True)

        # ---- OCR variant crops -------------------------------------------
        ocr_vehicles = debug_payload.get("ocr_variant_crops", [])
        if ocr_vehicles:
            st.markdown("**🔤 OCR Pre-processing Variants (per vehicle)**")
            for vehicle_entry in ocr_vehicles:
                vid = vehicle_entry.get("vehicle_track_id", "?")
                st.caption(f"Vehicle Track ID: {vid}")
                variants = vehicle_entry.get("variants", [])
                if not variants:
                    continue
                vcols = st.columns(len(variants))
                for col, variant in zip(vcols, variants):
                    vcrop = variant.get("crop")
                    if vcrop is None or vcrop.size == 0:
                        continue
                    display = vcrop if vcrop.ndim == 2 else cv2.cvtColor(vcrop, cv2.COLOR_BGR2RGB)
                    col.image(display, caption=variant.get("label", ""), use_container_width=True)


def initialize_engine_from_config(config: dict[str, Any]) -> Any | None:
    try:
        engine = load_enforcement_engine(
            primary_model_path=config["primary_model_path"],
            helmet_model_path=config["helmet_model_path"],
            seatbelt_model_path=config["seatbelt_model_path"],
            primary_confidence=config["primary_confidence"],
            helmet_confidence=config["helmet_confidence"],
            seatbelt_confidence=config["seatbelt_confidence"],
            device=config["device"],
        )
        reset_engine_runtime_state(engine)
        return engine
    except Exception as exc:
        st.error(f"Unable to initialize detection engine: {exc}")
        return None


def process_uploaded_image(
    frame: np.ndarray,
    engine: Any,
    config: dict[str, Any],
    frame_placeholder: Any,
    feed_placeholder: Any,
    metric_columns: list[Any],
    progress_placeholder: Any,
    debug_placeholder: Any,
    stop_line_polygon: list[tuple[float, float]] | None = None,
    no_parking_polygon: list[tuple[float, float]] | None = None,
) -> None:
    progress_bar = progress_placeholder.progress(0.0)
    annotated_frame, result, debug_payload = engine.process_frame(
        frame,
        debug_mode=config.get("dev_mode", False),
        stop_line_polygon_override=stop_line_polygon or None,
        no_parking_polygon_override=no_parking_polygon or None,
        manual_traffic_light=config.get("manual_traffic_light", "Auto-Detect"),
    )
    progress_bar.progress(1.0)

    # For single images there is no departure event, so force-flush all
    # active incidents immediately so challans are written to the database.
    final_challans = engine.flush_all_incidents()
    for challan in final_challans:
        msg = (
            f"[CHALLAN #{challan['id']:06d}] "
            f"Plate: {challan['license_plate']} | "
            f"{challan['violation_type']} | "
            f"PDF: {Path(challan['pdf_path']).name}"
        )
        st.session_state.live_alerts.append(msg)

    # Also surface any violations that were detected but didn't make it into
    # an incident (e.g. first-frame detections with no crop).
    for violation in result.get("violations", []):
        tid = violation.get("track_id", "?")
        vtype = violation.get("type", "Violation")
        conf = violation.get("confidence")
        conf_str = f" ({conf*100:.0f}%)" if conf is not None else ""
        alert_msg = f"[DETECTED] ID:{tid} — {vtype}{conf_str}"
        if alert_msg not in st.session_state.live_alerts:
            st.session_state.live_alerts.append(alert_msg)

    # Also show consolidated challans emitted by the engine during process_frame
    for challan in result.get("consolidated_challans", []):
        msg = (
            f"[CONSOLIDATED CHALLAN #{challan['id']:06d}] "
            f"Plate: {challan['license_plate']} | "
            f"{challan['violation_type']} | "
            f"PDF: {Path(challan['pdf_path']).name}"
        )
        if msg not in st.session_state.live_alerts:
            st.session_state.live_alerts.append(msg)

    frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
    feed_placeholder.empty()
    with feed_placeholder.container():
        render_activity_feed(st.session_state.live_alerts)

    tracked_objects = result.get("tracked_objects", {})
    metric_columns[0].metric("Frames", "1")
    metric_columns[1].metric("Vehicles", f"{tracked_objects.get('car', 0) + tracked_objects.get('motorcycle', 0):,}")
    metric_columns[2].metric("Persons", f"{tracked_objects.get('person', 0):,}")
    metric_columns[3].metric("Violations", f"{len(result.get('violations', [])):,}")

    if config.get("dev_mode") and debug_payload:
        with debug_placeholder.container():
            render_debug_expander(debug_payload, frame_number=1)

    st.success("Image analysis completed.")


def process_uploaded_video(
    uploaded_video: Any,
    engine: Any,
    config: dict[str, Any],
    frame_placeholder: Any,
    feed_placeholder: Any,
    metric_columns: list[Any],
    progress_placeholder: Any,
    debug_placeholder: Any,
    stop_line_polygon: list[tuple[float, float]] | None = None,
    no_parking_polygon: list[tuple[float, float]] | None = None,
) -> None:
    video_path = persist_uploaded_video(uploaded_video)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        st.error("Unable to read the uploaded video file.")
        video_path.unlink(missing_ok=True)
        return

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    processed_frames = 0
    frame_number = 0
    # Per-frame dedup so the live feed doesn't repeat the same violation
    # every single frame while a vehicle is still being tracked.
    live_feed_seen: set[tuple[int, str]] = set()
    run_started_at = time.time()
    progress_bar = progress_placeholder.progress(0.0)
    dev_mode = config.get("dev_mode", False)

    try:
        prev_alert_count = len(st.session_state.live_alerts)
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break

            frame_number += 1
            if config["max_frames"] and frame_number > config["max_frames"]:
                break
            if frame_number % config["frame_stride"] != 0:
                continue

            annotated_frame, result, debug_payload = engine.process_frame(
                frame,
                debug_mode=dev_mode,
                stop_line_polygon_override=stop_line_polygon or None,
                no_parking_polygon_override=no_parking_polygon or None,
                manual_traffic_light=config.get("manual_traffic_light", "Auto-Detect"),
            )
            processed_frames += 1

            # ── Live feed: first detection per (track_id, type) ──────────
            for violation in result.get("violations", []):
                alert_key = (int(violation.get("track_id", -1)), str(violation.get("type", "")))
                if alert_key in live_feed_seen:
                    continue
                live_feed_seen.add(alert_key)
                tid = violation.get("track_id", "?")
                vtype = violation.get("type", "Violation")
                conf = violation.get("confidence")
                conf_str = f" ({conf*100:.0f}%)" if conf is not None else ""
                st.session_state.live_alerts.append(
                    f"[LIVE] ID:{tid} — {vtype}{conf_str} (challan pending departure)"
                )

            # ── Consolidated challans emitted by engine on departure ──────
            for challan in result.get("consolidated_challans", []):
                msg = (
                    f"[CHALLAN #{challan['id']:06d}] "
                    f"Plate: {challan['license_plate']} | "
                    f"{challan['violation_type']} | "
                    f"PDF: {Path(challan['pdf_path']).name}"
                )
                st.session_state.live_alerts.append(msg)

            # Throttle UI updates to every 5 processed frames to prevent layout jitter.
            # The inference still runs every frame; only the display is throttled.
            if processed_frames % 5 == 0 or processed_frames == 1:
                frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)

                elapsed = max(time.time() - run_started_at, 1e-6)
                current_fps = processed_frames / elapsed
                metric_columns[0].metric("Frame", f"{frame_number:,}")
                metric_columns[1].metric("Processed", f"{processed_frames:,}")
                metric_columns[2].metric("FPS", f"{current_fps:.2f}")
                metric_columns[3].metric("Alerts", f"{len(st.session_state.live_alerts):,}")

                if total_frames > 0:
                    progress_bar.progress(min(frame_number / total_frames, 1.0))

            # Only re-render the live feed when new alerts arrive
            current_alert_count = len(st.session_state.live_alerts)
            if current_alert_count != prev_alert_count:
                feed_placeholder.empty()
                with feed_placeholder.container():
                    render_activity_feed(st.session_state.live_alerts)
                prev_alert_count = current_alert_count

            if dev_mode and debug_payload and processed_frames % 5 == 0:
                debug_placeholder.empty()
                with debug_placeholder.container():
                    render_debug_expander(debug_payload, frame_number=frame_number)

        final_challans = engine.flush_all_incidents()
        for challan in final_challans:
            msg = (
                f"[CHALLAN #{challan['id']:06d}] "
                f"Plate: {challan['license_plate']} | "
                f"{challan['violation_type']} | "
                f"PDF: {Path(challan['pdf_path']).name}"
            )
            st.session_state.live_alerts.append(msg)
            
        with feed_placeholder.container():
            render_activity_feed(st.session_state.live_alerts)

        progress_bar.progress(1.0)
        st.success("Video analysis completed.")
    finally:
        capture.release()
        video_path.unlink(missing_ok=True)


def render_evidence_processing(config: dict[str, Any]) -> None:
    st.title("Evidence Image or Video Analysis")

    uploaded_file = st.file_uploader(
        "Upload traffic evidence",
        type=IMAGE_UPLOAD_TYPES + VIDEO_UPLOAD_TYPES,
        help="Supported images: JPG, JPEG, PNG, BMP, WEBP. Supported videos: MP4, AVI.",
    )
    if uploaded_file is None:
        st.info("Upload a traffic image or video file to start analysis.")
        return

    image_mode = is_image_upload(uploaded_file)
    decoded_image = decode_uploaded_image(uploaded_file) if image_mode else None

    # ── ROI Calibration (video only) ──────────────────────────────────────
    stop_line_polygon: list[tuple[float, float]] | None = None
    no_parking_polygon: list[tuple[float, float]] | None = None

    if not image_mode:
        reference_frame = extract_first_frame(uploaded_file)
        if reference_frame is not None:
            render_roi_calibration_ui(reference_frame)
            # Collect committed polygons (only pass when all 4 points are defined)
            raw_stop  = [pt for pt in st.session_state.get("custom_roi_stop", []) if pt is not None]
            raw_park  = [pt for pt in st.session_state.get("custom_roi_parking", []) if pt is not None]
            if len(raw_stop) == 4:
                stop_line_polygon = [(float(x), float(y)) for x, y in raw_stop]
            if len(raw_park) == 4:
                no_parking_polygon = [(float(x), float(y)) for x, y in raw_park]
        else:
            st.warning("Could not extract a reference frame from the video for ROI calibration.")

    left_column, right_column = st.columns([3.2, 1.1], gap="large")
    with left_column:
        frame_placeholder = st.empty()
        if image_mode:
            if decoded_image is None:
                st.error("Unable to decode the uploaded image.")
                return
            frame_placeholder.image(decoded_image, channels="BGR", use_container_width=True)

        progress_placeholder = st.empty()
        metric_columns = st.columns(4)
        button_label = "Analyze Image" if image_mode else "Start Video Analysis"
        start_analysis = st.button(button_label, type="primary", use_container_width=True)
        debug_placeholder = st.empty()  # Dev Mode panel sits directly below the player

    with right_column:
        st.subheader("Live Activity")
        feed_placeholder = st.empty()
        render_activity_feed(st.session_state.live_alerts)

    if not start_analysis:
        return

    engine = initialize_engine_from_config(config)
    if engine is None:
        return

    if image_mode:
        process_uploaded_image(
            frame=decoded_image,
            engine=engine,
            config=config,
            frame_placeholder=frame_placeholder,
            feed_placeholder=feed_placeholder,
            metric_columns=metric_columns,
            progress_placeholder=progress_placeholder,
            debug_placeholder=debug_placeholder,
            stop_line_polygon=stop_line_polygon,
            no_parking_polygon=no_parking_polygon,
        )
    else:
        process_uploaded_video(
            uploaded_video=uploaded_file,
            engine=engine,
            config=config,
            frame_placeholder=frame_placeholder,
            feed_placeholder=feed_placeholder,
            metric_columns=metric_columns,
            progress_placeholder=progress_placeholder,
            debug_placeholder=debug_placeholder,
            stop_line_polygon=stop_line_polygon,
            no_parking_polygon=no_parking_polygon,
        )


def load_violation_records(db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    initialize_database(db_path)
    with sqlite3.connect(db_path) as connection:
        records = pd.read_sql_query(
            """
            SELECT id, timestamp, violation_type, confidence, license_plate, crop_path, pdf_path
            FROM challans
            ORDER BY timestamp DESC, id DESC
            """,
            connection,
        )

    if records.empty:
        return records

    records["timestamp_dt"] = pd.to_datetime(records["timestamp"], errors="coerce")
    records["confidence"] = pd.to_numeric(records["confidence"], errors="coerce").fillna(0.0)
    return records


def render_violation_type_chart(records: pd.DataFrame) -> None:
    distribution = (
        records.groupby("violation_type", dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    figure = px.bar(
        distribution,
        x="violation_type",
        y="count",
        text="count",
        labels={"violation_type": "Violation Type", "count": "Count"},
    )
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10), xaxis_tickangle=-25)
    st.plotly_chart(figure, use_container_width=True)


def render_timeline_chart(records: pd.DataFrame) -> None:
    timeline_source = records.dropna(subset=["timestamp_dt"]).copy()
    if timeline_source.empty:
        st.info("No timestamped records available for timeline analysis.")
        return

    timeline = (
        timeline_source.set_index("timestamp_dt")
        .resample("D")
        .size()
        .reset_index(name="count")
    )
    figure = px.line(
        timeline,
        x="timestamp_dt",
        y="count",
        markers=True,
        labels={"timestamp_dt": "Date", "count": "Violations"},
    )
    figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(figure, use_container_width=True)


def filter_records(records: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query.strip() or records.empty:
        return records

    query = query.strip().lower()
    searchable_columns = ["violation_type", "license_plate", "timestamp", "pdf_path", "crop_path"]
    mask = pd.Series(False, index=records.index)
    for column in searchable_columns:
        mask = mask | records[column].astype(str).str.lower().str.contains(query, na=False)
    return records[mask]


def render_pdf_download(filtered_records: pd.DataFrame) -> None:
    downloadable = filtered_records[filtered_records["pdf_path"].astype(str).str.len() > 0]
    if downloadable.empty:
        st.warning("No generated PDF challans are available for the current table selection.")
        return

    selected_id = st.selectbox(
        "Select Challan PDF",
        options=downloadable["id"].tolist(),
        format_func=lambda challan_id: f"Challan #{int(challan_id):06d}",
    )
    selected_row = downloadable[downloadable["id"] == selected_id].iloc[0]
    pdf_path = Path(str(selected_row["pdf_path"]))

    if not pdf_path.exists():
        st.warning(f"PDF file not found on disk: {pdf_path}")
        return

    st.download_button(
        "Download Selected E-Challan PDF",
        data=pdf_path.read_bytes(),
        file_name=pdf_path.name,
        mime="application/pdf",
        use_container_width=True,
    )


def render_analytics_dashboard() -> None:
    st.title("System Performance Analytics Dashboard")

    records = load_violation_records()
    if records.empty:
        st.info("No violation records found in violations.db.")
        return

    summary_columns = st.columns(4)
    summary_columns[0].metric("Total Challans", f"{len(records):,}")
    summary_columns[1].metric("Violation Types", f"{records['violation_type'].nunique():,}")
    summary_columns[2].metric("Unique Plates", f"{records['license_plate'].nunique():,}")
    summary_columns[3].metric("Average Confidence", f"{records['confidence'].mean() * 100:.2f}%")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.subheader("Violation Distribution")
        render_violation_type_chart(records)
    with chart_right:
        st.subheader("Violation Timeline")
        render_timeline_chart(records)

    st.subheader("Violation Records")
    query = st.text_input("Search records", placeholder="Search by plate, violation type, timestamp, or file path")
    filtered_records = filter_records(records, query)

    display_columns = ["id", "timestamp", "violation_type", "confidence", "license_plate", "crop_path", "pdf_path"]
    display_frame = filtered_records[display_columns].copy()
    display_frame["confidence"] = (display_frame["confidence"] * 100).round(2).astype(str) + "%"

    st.dataframe(
        display_frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": st.column_config.NumberColumn("ID", format="%d"),
            "timestamp": st.column_config.TextColumn("Timestamp"),
            "violation_type": st.column_config.TextColumn("Violation Type"),
            "confidence": st.column_config.TextColumn("Confidence"),
            "license_plate": st.column_config.TextColumn("License Plate"),
            "crop_path": st.column_config.TextColumn("Image Evidence"),
            "pdf_path": st.column_config.TextColumn("PDF Challan"),
        },
    )
    render_pdf_download(filtered_records)


def main() -> None:
    configure_page()
    # Core session state
    if "live_alerts" not in st.session_state:
        st.session_state.live_alerts = []
    # HITL ROI calibration state
    if "custom_roi_stop" not in st.session_state:
        st.session_state.custom_roi_stop = []
    if "custom_roi_parking" not in st.session_state:
        st.session_state.custom_roi_parking = []

    config = render_sidebar()
    if config["operation_mode"] == "Evidence Image or Video Analysis":
        render_evidence_processing(config)
    else:
        render_analytics_dashboard()


if __name__ == "__main__":
    main()
