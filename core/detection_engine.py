from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Iterable, Sequence

import cv2
import numpy as np
from ultralytics import YOLO

from utils.preprocessing import preprocess_frame


LOGGER = logging.getLogger(__name__)

BBox = tuple[int, int, int, int]
Point = tuple[float, float]
Polygon = Sequence[Point]


@dataclass(frozen=True)
class TrackDetection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    centroid: Point


@dataclass
class ViolationRecord:
    track_id: int
    violation_type: str
    bbox: BBox
    confidence: float | None = None
    related_track_ids: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "type": self.violation_type,
            "bbox": [int(value) for value in self.bbox],
            "confidence": None if self.confidence is None else round(float(self.confidence), 4),
            "related_track_ids": [int(track_id) for track_id in self.related_track_ids],
            "metadata": self.metadata,
        }


class TrafficEnforcementEngine:
    """
    End-to-end frame processor for traffic detection, tracking, and violation logic.

    Camera geometry is intentionally configurable because lane direction, stop lines,
    and no-parking zones vary by installation. Polygon coordinates may be absolute
    pixels or normalized points in the [0, 1] range.
    """

    PERSON_LABELS = {"person"}
    MOTORCYCLE_LABELS = {"motorcycle", "motorbike", "bike"}
    CAR_LABELS = {"car"}
    TRAFFIC_LIGHT_LABELS = {"traffic light", "traffic_light", "traffic-light"}

    DEFAULT_NO_HELMET_LABELS = {"no-helmet", "no helmet", "no_helmet", "without-helmet"}
    DEFAULT_NO_SEATBELT_LABELS = {
        "person-noseatbelt",
        "person noseatbelt",
        "person_noseatbelt",
        "no-seatbelt",
        "no seatbelt",
        "no_seatbelt",
        "without-seatbelt",
    }

    VIOLATION_COLORS = {
        "Triple Riding Violation": (0, 165, 255),
        "Helmet Violation": (0, 0, 255),
        "Seatbelt Violation": (255, 0, 0),
        "Wrong-side Violation": (255, 0, 255),
        "Red-light / Stop-line Violation": (0, 255, 255),
        "Illegal Parking Violation": (128, 0, 255),
    }

    DEFAULT_BOX_COLOR = (0, 180, 0)
    STOP_LINE_COLOR = (0, 255, 255)
    NO_PARKING_COLOR = (128, 0, 255)

    def __init__(
        self,
        primary_model_path: str | Path = "yolov8s.pt",
        helmet_model_path: str | Path = "models/helmet_best.pt",
        seatbelt_model_path: str | Path = "models/seatbelt_best.pt",
        tracker_config: str = "bytetrack.yaml",
        device: str | int | None = None,
        image_size: int = 640,
        primary_conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        helmet_conf_threshold: float = 0.55,
        seatbelt_conf_threshold: float = 0.50,
        allowed_lane_vector: Point = (0.0, 1.0),
        wrong_side_dot_threshold: float = -0.65,
        wrong_side_min_displacement: float = 20.0,
        stop_line_polygon: Polygon | None = None,
        no_parking_polygon: Polygon | None = None,
        traffic_light_rois: Sequence[Polygon] | None = None,
        red_light_min_ratio: float = 0.025,
        parking_stationary_pixels: float = 5.0,
        parking_stationary_frames: int = 200,
        preprocessing_mode: str = "auto",
    ) -> None:
        self.primary_model = YOLO(str(primary_model_path))
        self.helmet_model = self._load_local_model(helmet_model_path, "helmet")
        self.seatbelt_model = self._load_local_model(seatbelt_model_path, "seatbelt")

        self.tracker_config = tracker_config
        self.device = device
        self.image_size = image_size
        self.primary_conf_threshold = primary_conf_threshold
        self.iou_threshold = iou_threshold
        self.helmet_conf_threshold = helmet_conf_threshold
        self.seatbelt_conf_threshold = seatbelt_conf_threshold
        self.wrong_side_dot_threshold = wrong_side_dot_threshold
        self.wrong_side_min_displacement = wrong_side_min_displacement
        self.red_light_min_ratio = red_light_min_ratio
        self.parking_stationary_pixels = parking_stationary_pixels
        self.parking_stationary_frames = parking_stationary_frames
        self.preprocessing_mode = preprocessing_mode

        self.allowed_lane_vector = self._normalize_vector(allowed_lane_vector)
        self.stop_line_polygon = stop_line_polygon
        self.no_parking_polygon = no_parking_polygon
        self.traffic_light_rois = list(traffic_light_rois or [])

        self.frame_index = 0
        self.centroid_history: dict[int, Deque[Point]] = defaultdict(lambda: deque(maxlen=30))
        self.parking_state: dict[int, Deque[Point]] = defaultdict(
            lambda: deque(maxlen=self.parking_stationary_frames)
        )

        # ------------------------------------------------------------------
        # Consolidated incident tracking
        # active_incidents[track_id] accumulates per-vehicle violations until
        # Challan is committed to DB after the vehicle departs (absent for
        # > INCIDENT_TTL_FRAMES) OR when flush_all_incidents() is called at end
        # of video.  Lowered to 3 frames so short clips get challans quickly.
        # ------------------------------------------------------------------
        self.INCIDENT_TTL_FRAMES: int = 3
        self.active_incidents: dict[int, dict[str, Any]] = {}
        self.incident_last_seen: dict[int, int] = {}  # track_id -> last frame_index

    @staticmethod
    def _resolve_weights_path(weights_path: str | Path) -> Path:
        weights = Path(weights_path)
        if weights.exists():
            return weights

        project_root = Path(__file__).resolve().parent.parent
        root_fallback = project_root / weights.name
        if root_fallback.exists():
            return root_fallback

        return weights

    @classmethod
    def _load_local_model(cls, weights_path: str | Path, model_label: str) -> YOLO:
        weights = cls._resolve_weights_path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"Fine-tuned {model_label} weights not found: {weights}. "
                "Train the model first or pass the correct weights path."
            )
        return YOLO(str(weights))

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None:
            raise ValueError("frame must not be None")
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a numpy.ndarray")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image with shape (height, width, 3)")
        if frame.size == 0:
            raise ValueError("frame must not be empty")

    @staticmethod
    def _normalize_vector(vector: Point) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        if norm <= 0:
            raise ValueError("allowed_lane_vector must be non-zero")
        return array / norm

    @staticmethod
    def _clip_bbox(bbox: Sequence[float], frame_shape: tuple[int, ...]) -> BBox:
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width - 1))
        y2 = max(0, min(int(round(y2)), height - 1))
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_centroid(bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    @staticmethod
    def _bbox_bottom_center(bbox: BBox) -> Point:
        x1, _, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y2))

    @staticmethod
    def _bbox_area(bbox: BBox) -> float:
        x1, y1, x2, y2 = bbox
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    @staticmethod
    def _bbox_intersection_area(first: BBox, second: BBox) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        return float(max(0, x2 - x1) * max(0, y2 - y1))

    def _bbox_iou(self, first: BBox, second: BBox) -> float:
        intersection = self._bbox_intersection_area(first, second)
        first_area = self._bbox_area(first)
        second_area = self._bbox_area(second)
        union = first_area + second_area - intersection
        if union <= 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def _point_inside_bbox(point: Point, bbox: BBox) -> bool:
        x, y = point
        return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]

    @staticmethod
    def _label_key(label: str) -> str:
        return label.strip().lower().replace("_", " ").replace("-", " ")

    def _is_person_on_motorcycle(self, person: TrackDetection, motorcycle: TrackDetection) -> bool:
        person_area = self._bbox_area(person.bbox)
        if person_area <= 0:
            return False

        px1, py1, px2, py2 = person.bbox
        mx1, my1, mx2, my2 = motorcycle.bbox
        motorcycle_width = max(1, mx2 - mx1)
        motorcycle_height = max(1, my2 - my1)

        expanded_motorcycle = (
            int(mx1 - motorcycle_width * 0.20),
            int(my1 - motorcycle_height * 0.35),
            int(mx2 + motorcycle_width * 0.20),
            int(my2 + motorcycle_height * 0.20),
        )

        person_width = max(1, px2 - px1)
        person_height = max(1, py2 - py1)
        lower_body_bbox = (
            px1,
            int(py1 + person_height * 0.55),
            px2,
            py2,
        )
        lower_body_area = self._bbox_area(lower_body_bbox)
        lower_body_intersection = self._bbox_intersection_area(lower_body_bbox, expanded_motorcycle)
        lower_body_overlap = lower_body_intersection / lower_body_area if lower_body_area > 0 else 0.0

        person_bottom_center = (float(px1 + px2) / 2.0, float(py2))
        horizontal_overlap = max(0, min(px2, mx2) - max(px1, mx1)) / float(person_width)
        bottom_near_motorcycle = (my1 - motorcycle_height * 0.45) <= py2 <= (my2 + motorcycle_height * 0.55)

        return (
            self._point_inside_bbox(person_bottom_center, expanded_motorcycle)
            and bottom_near_motorcycle
            and (lower_body_overlap >= 0.25 or horizontal_overlap >= 0.35)
        )

    def _nms_person_tracks(self, person_tracks: Sequence[TrackDetection], iou_threshold: float = 0.40) -> list[TrackDetection]:
        """
        Remove duplicate person detections before rider counting.

        Person boxes with IoU greater than iou_threshold are treated as the same
        physical rider and only the highest-confidence detection is retained.
        """
        sorted_tracks = sorted(person_tracks, key=lambda track: track.confidence, reverse=True)
        selected_tracks: list[TrackDetection] = []

        for candidate in sorted_tracks:
            is_duplicate = any(
                self._bbox_iou(candidate.bbox, selected.bbox) > iou_threshold
                for selected in selected_tracks
            )
            if not is_duplicate:
                selected_tracks.append(candidate)

        return selected_tracks

    def _classify_riders_by_position(
        self,
        riders: list[TrackDetection],
        motorcycle: TrackDetection,
    ) -> list[tuple[str, TrackDetection]]:
        """
        Label each rider relative to the motorcycle's direction of travel.

        Since the camera is typically mounted overhead / at an angle looking
        down the road, a lower Y centroid (closer to the top of the frame)
        means the person is *ahead* in the direction of travel. We therefore
        sort ascending by centroid Y and assign:
          - index 0  → 'Rider'   (front / driver)
          - index 1  → 'Pillion' (immediately behind)
          - index 2+ → 'Pillion {n}' for triple-ride edge cases

        Returns a list of (role, TrackDetection) tuples in front-to-back order.
        """
        sorted_riders = sorted(riders, key=lambda r: r.centroid[1])  # ascending Y
        labeled: list[tuple[str, TrackDetection]] = []
        for idx, rider in enumerate(sorted_riders):
            if idx == 0:
                role = "Rider"
            elif idx == 1:
                role = "Pillion"
            else:
                role = f"Pillion {idx}"
            labeled.append((role, rider))
        return labeled

    def _update_incident(
        self,
        track_id: int,
        violation: ViolationRecord,
        frame: np.ndarray,
        best_crop: np.ndarray | None,
    ) -> None:
        """
        Merge a new ViolationRecord into the active_incidents store.
        Keeps the largest (best-area) crop seen for this track_id.
        """
        if track_id not in self.active_incidents:
            self.active_incidents[track_id] = {
                "violations": [],
                "seen_types": set(),
                "best_crop": None,
                "plate_text": None,
            }

        inc = self.active_incidents[track_id]

        # Deduplicate violations by type within the same incident
        if violation.violation_type not in inc["seen_types"]:
            inc["seen_types"].add(violation.violation_type)
            inc["violations"].append({
                "type":       violation.violation_type,
                "confidence": violation.confidence or 0.0,
            })

        # Keep the largest crop (most context)
        if best_crop is not None and best_crop.size > 0:
            existing = inc["best_crop"]
            if existing is None or best_crop.size > existing.size:
                inc["best_crop"] = best_crop.copy()

    def _flush_departed_incidents(
        self,
        active_track_ids: set[int],
        plate_text_by_track: dict[int, str],
        db_path: Any = None,
        force_all: bool = False,
    ) -> list[dict[str, Any]]:
        """
        For every tracked vehicle that has been absent for > INCIDENT_TTL_FRAMES,
        emit a consolidated challan and remove it from active_incidents.

        Returns a list of challan dicts (one per departed vehicle).
        """
        from utils.evidence_generator import DEFAULT_DB_PATH, save_violation_record

        departed = [
            tid for tid, last_seen in self.incident_last_seen.items()
            if tid in self.active_incidents
            and (force_all or (tid not in active_track_ids and (self.frame_index - last_seen) > self.INCIDENT_TTL_FRAMES))
        ]

        challans: list[dict[str, Any]] = []
        for tid in departed:
            inc = self.active_incidents.get(tid)
            if not inc or not inc["violations"]:
                self.active_incidents.pop(tid, None)
                self.incident_last_seen.pop(tid, None)
                continue

            crop = inc.get("best_crop")
            plate = plate_text_by_track.get(tid) or inc.get("plate_text")

            if crop is None or crop.size == 0:
                # No crop available – skip PDF generation, clean up
                self.active_incidents.pop(tid, None)
                self.incident_last_seen.pop(tid, None)
                continue

            try:
                record = save_violation_record(
                    violations=inc["violations"],
                    license_plate=plate,
                    frame_crop=crop,
                    db_path=db_path or DEFAULT_DB_PATH,
                )
                challans.append(record)
                LOGGER.info(
                    "Consolidated challan #%s issued for Track ID %s (%d violations)",
                    record["id"], tid, len(inc["violations"]),
                )
            except Exception as exc:
                LOGGER.warning("Failed to generate challan for Track ID %s: %s", tid, exc)
            finally:
                self.active_incidents.pop(tid, None)
                self.incident_last_seen.pop(tid, None)

        return challans

    def flush_all_incidents(self, db_path: Any = None) -> list[dict[str, Any]]:
        """
        Force-flush all active incidents and generate consolidated challans
        for any vehicles still being tracked. Use this at the end of a video stream.
        """
        return self._flush_departed_incidents(
            active_track_ids=set(),
            plate_text_by_track={},
            db_path=db_path,
            force_all=True,
        )

    # ------------------------------------------------------------------
    # Geometric Triple Riding detection
    # ------------------------------------------------------------------

    # Horizontal padding (px) added to each side of a motorcycle bbox
    # to account for wide seating arrangements.
    _MOTO_EXPAND_PX: int = 10

    # Two person centroids closer than this (px) are considered the same
    # physical person and de-duplicated before counting riders.
    _CENTROID_MERGE_PX: float = 15.0

    def _triple_riding_geometric(
        self,
        motorcycle: TrackDetection,
        persons: list[TrackDetection],
    ) -> tuple[bool, int, list[tuple[float, float]]]:
        """
        Detect Triple Riding using centroid-based geometric filtering.

        Algorithm
        ---------
        1. Expand the motorcycle bbox horizontally by *_MOTO_EXPAND_PX* on each
           side to handle wide seating.
        2. Collect all person centroids that fall strictly inside that boundary.
        3. De-duplicate: merge any pair of centroids closer than *_CENTROID_MERGE_PX*
           pixels (keep the first one seen – highest-confidence order preserved by
           caller's sort).
        4. If de-duplicated count >= 3: Triple Riding Violation.

        Returns
        -------
        is_violation : bool
        unique_rider_count : int
        unique_centroids : list of (x, y) that passed de-duplication
        """
        if not persons or motorcycle is None:
            return False, 0, []

        mx1, my1, mx2, my2 = motorcycle.bbox
        exp_bbox: BBox = (
            max(0, mx1 - self._MOTO_EXPAND_PX),
            my1,
            mx2 + self._MOTO_EXPAND_PX,
            my2,
        )

        # Collect centroids inside the expanded boundary, highest confidence first
        sorted_persons = sorted(persons, key=lambda p: p.confidence, reverse=True)
        candidates: list[tuple[float, float]] = [
            (p.centroid[0], p.centroid[1])
            for p in sorted_persons
            if self._point_inside_bbox(p.centroid, exp_bbox)
        ]

        if not candidates:
            return False, 0, []

        # Internal NMS: merge centroids within _CENTROID_MERGE_PX
        unique: list[tuple[float, float]] = []
        for cx, cy in candidates:
            is_duplicate = any(
                ((cx - ux) ** 2 + (cy - uy) ** 2) ** 0.5 < self._CENTROID_MERGE_PX
                for ux, uy in unique
            )
            if not is_duplicate:
                unique.append((cx, cy))

        return len(unique) >= 3, len(unique), unique

    def _triple_riding_helmet_count(
        self,
        frame: np.ndarray,
        motorcycle: TrackDetection,
    ) -> tuple[bool, int]:
        """
        Cross-reference Triple Riding by counting Helmet + No-Helmet detections
        in the upper 40% of the motorcycle bounding box.

        If the helmet model finds >= 3 head objects in that region, it is
        virtually certain three people are seated on the motorcycle, regardless
        of whether their full-body centroids were detected.

        Returns
        -------
        is_violation : bool
        head_count   : int  – total Helmet + No-Helmet detections in the crop
        """
        mx1, my1, mx2, my2 = motorcycle.bbox
        head_zone_h = int((my2 - my1) * 0.40)
        head_zone_bbox: BBox = (mx1, my1, mx2, my1 + max(head_zone_h, 1))
        head_crop = self._safe_crop(frame, head_zone_bbox)

        if head_crop is None or head_crop.size == 0 or min(head_crop.shape[:2]) < 8:
            return False, 0

        try:
            results = self.helmet_model.predict(
                head_crop, conf=self.helmet_conf_threshold, **self._prediction_kwargs()
            )
        except Exception as exc:
            LOGGER.debug("Helmet-count cross-reference failed: %s", exc)
            return False, 0

        head_count = 0
        for res in results:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            names = getattr(res, "names", {}) or {}
            for cls_id in boxes.cls.cpu().numpy().astype(int):
                label = self._label_key(str(names.get(int(cls_id), cls_id)))
                # Count both helmet-wearing and no-helmet heads
                if "helmet" in label or "head" in label:
                    head_count += 1

        return head_count >= 3, head_count

    def _default_stop_line_polygon(self, frame_shape: tuple[int, ...]) -> np.ndarray:
        height, width = frame_shape[:2]
        return np.asarray(
            [
                (0.10 * width, 0.62 * height),
                (0.90 * width, 0.62 * height),
                (0.90 * width, 0.68 * height),
                (0.10 * width, 0.68 * height),
            ],
            dtype=np.float32,
        )

    def _default_no_parking_polygon(self, frame_shape: tuple[int, ...]) -> np.ndarray:
        height, width = frame_shape[:2]
        return np.asarray(
            [
                (0.65 * width, 0.52 * height),
                (0.98 * width, 0.52 * height),
                (0.98 * width, 0.98 * height),
                (0.65 * width, 0.98 * height),
            ],
            dtype=np.float32,
        )

    def _resolve_polygon(
        self,
        polygon: Polygon | np.ndarray | None,
        frame_shape: tuple[int, ...],
        default_polygon: np.ndarray,
    ) -> np.ndarray:
        points = np.asarray(default_polygon if polygon is None else polygon, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("polygon must contain points shaped as (x, y)")

        if points.size > 0 and np.all(points >= 0.0) and np.all(points <= 1.0):
            height, width = frame_shape[:2]
            points = points * np.asarray([width, height], dtype=np.float32)

        return points.astype(np.int32)

    @staticmethod
    def _point_inside_polygon(point: Point, polygon: np.ndarray) -> bool:
        return cv2.pointPolygonTest(polygon.astype(np.float32), point, False) >= 0

    @staticmethod
    def _orientation(a: Point, b: Point, c: Point) -> int:
        value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if abs(value) < 1e-6:
            return 0
        return 1 if value > 0 else 2

    @staticmethod
    def _on_segment(a: Point, b: Point, c: Point) -> bool:
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    def _segments_intersect(self, a: Point, b: Point, c: Point, d: Point) -> bool:
        o1 = self._orientation(a, b, c)
        o2 = self._orientation(a, b, d)
        o3 = self._orientation(c, d, a)
        o4 = self._orientation(c, d, b)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and self._on_segment(a, c, b):
            return True
        if o2 == 0 and self._on_segment(a, d, b):
            return True
        if o3 == 0 and self._on_segment(c, a, d):
            return True
        if o4 == 0 and self._on_segment(c, b, d):
            return True
        return False

    def _segment_crosses_polygon(self, start: Point, end: Point, polygon: np.ndarray) -> bool:
        if self._point_inside_polygon(end, polygon):
            return True

        points = [(float(x), float(y)) for x, y in polygon]
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            if self._segments_intersect(start, end, point, next_point):
                return True
        return False

    def _extract_tracks(self, result: Any, frame_shape: tuple[int, ...]) -> list[TrackDetection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = getattr(result, "names", {}) or {}
        xyxy = boxes.xyxy.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

        tracks: list[TrackDetection] = []
        for index, bbox_values in enumerate(xyxy):
            bbox = self._clip_bbox(bbox_values, frame_shape)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            class_id = int(class_ids[index])
            class_name = str(names.get(class_id, class_id))
            track_id = int(track_ids[index]) if track_ids is not None else -(index + 1)

            tracks.append(
                TrackDetection(
                    track_id=track_id,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[index]),
                    bbox=bbox,
                    centroid=self._bbox_centroid(bbox),
                )
            )

        return tracks

    def _class_matches(self, detection: TrackDetection, labels: set[str]) -> bool:
        return self._label_key(detection.class_name) in {self._label_key(label) for label in labels}

    def _filter_tracks(self, tracks: Iterable[TrackDetection], labels: set[str]) -> list[TrackDetection]:
        return [track for track in tracks if self._class_matches(track, labels)]

    def _safe_crop(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        x1, y1, x2, y2 = self._clip_bbox(bbox, frame.shape)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _crop(self, frame: np.ndarray, bbox: BBox) -> np.ndarray | None:
        return self._safe_crop(frame, bbox)

    def _helmet_crop(self, frame: np.ndarray, person_bbox: BBox) -> np.ndarray | None:
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1
        helmet_bbox = (
            int(x1),
            int(y1),
            int(x2),
            int(y1 + int(height * 0.3)),
        )
        return self._safe_crop(frame, helmet_bbox)

    def _seatbelt_crop(self, frame: np.ndarray, car_bbox: BBox) -> np.ndarray | None:
        x1, y1, x2, y2 = car_bbox
        height = y2 - y1
        seatbelt_bbox = (
            int(x1),
            int(y1),
            int(x2),
            int(y1 + int(height * 0.5)),
        )
        return self._safe_crop(frame, seatbelt_bbox)

    def _prediction_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"imgsz": self.image_size, "verbose": False}
        if self.device is not None:
            kwargs["device"] = self.device
        return kwargs

    def _detect_target_label(
        self,
        model: YOLO,
        crop: np.ndarray | None,
        threshold: float,
        target_labels: set[str],
        semantic_tokens: tuple[str, ...],
    ) -> float | None:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 8:
            return None

        results = model.predict(crop, conf=threshold, **self._prediction_kwargs())
        best_confidence: float | None = None

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            names = getattr(result, "names", {}) or {}
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for class_id, confidence in zip(class_ids, confidences):
                raw_label = str(names.get(int(class_id), class_id))
                normalized_label = self._label_key(raw_label)
                normalized_targets = {self._label_key(label) for label in target_labels}

                label_is_target = normalized_label in normalized_targets
                label_has_tokens = all(token in normalized_label for token in semantic_tokens)

                if label_is_target or label_has_tokens:
                    best_confidence = max(float(confidence), best_confidence or 0.0)

        return best_confidence

    def _detect_no_helmet(self, frame: np.ndarray, person: TrackDetection) -> float | None:
        head_crop = self._helmet_crop(frame, person.bbox)
        return self._detect_target_label(
            model=self.helmet_model,
            crop=head_crop,
            threshold=self.helmet_conf_threshold,
            target_labels=self.DEFAULT_NO_HELMET_LABELS,
            semantic_tokens=("no", "helmet"),
        )

    def _detect_no_seatbelt(self, frame: np.ndarray, car: TrackDetection) -> float | None:
        seatbelt_crop = self._seatbelt_crop(frame, car.bbox)
        return self._detect_target_label(
            model=self.seatbelt_model,
            crop=seatbelt_crop,
            threshold=self.seatbelt_conf_threshold,
            target_labels=self.DEFAULT_NO_SEATBELT_LABELS,
            semantic_tokens=("no", "seatbelt"),
        )

    def _traffic_light_state_from_crop(self, crop: np.ndarray | None) -> tuple[str, dict[str, Any]]:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 4:
            return "UNKNOWN", {}

        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
        blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
        height = blurred.shape[0]
        max_brightness = float(np.max(blurred))

        if max_brightness <= 0:
            return "UNKNOWN", {"max_brightness": max_brightness}

        bright_threshold = max(220.0, max_brightness * 0.90)
        bright_pixels = np.argwhere(blurred >= bright_threshold)
        if bright_pixels.size == 0:
            _, _, _, max_location = cv2.minMaxLoc(blurred)
            bright_y = float(max_location[1])
        else:
            bright_y = float(np.mean(bright_pixels[:, 0]))

        top_third_limit = height / 3.0
        light_state = "RED" if bright_y < top_third_limit else "NOT_RED"
        return light_state, {
            "brightest_region_y": round(bright_y, 2),
            "top_third_limit_y": round(top_third_limit, 2),
            "max_brightness": round(max_brightness, 2),
        }

    def _polygon_crop(self, frame: np.ndarray, polygon: np.ndarray) -> np.ndarray | None:
        x, y, width, height = cv2.boundingRect(polygon.astype(np.int32))
        if width <= 0 or height <= 0:
            return None

        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            return None

        shifted_polygon = polygon.copy()
        shifted_polygon[:, 0] -= x
        shifted_polygon[:, 1] -= y
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [shifted_polygon.astype(np.int32)], 255)
        return cv2.bitwise_and(crop, crop, mask=mask)

    def _detect_traffic_light_state(
        self,
        frame: np.ndarray,
        all_tracks: Sequence[TrackDetection],
        frame_shape: tuple[int, ...],
    ) -> tuple[str, dict[str, Any]]:
        traffic_lights = self._filter_tracks(all_tracks, self.TRAFFIC_LIGHT_LABELS)
        first_detected_state: tuple[str, dict[str, Any]] | None = None

        for light in traffic_lights:
            light_state, metadata = self._traffic_light_state_from_crop(self._crop(frame, light.bbox))
            metadata.update({"traffic_light_track_id": light.track_id, "traffic_light_bbox": light.bbox})
            if light_state == "RED":
                return light_state, metadata
            if light_state != "UNKNOWN" and first_detected_state is None:
                first_detected_state = (light_state, metadata)

        for roi in self.traffic_light_rois:
            polygon = self._resolve_polygon(roi, frame_shape, np.asarray(roi, dtype=np.float32))
            light_state, metadata = self._traffic_light_state_from_crop(self._polygon_crop(frame, polygon))
            metadata.update({"traffic_light_roi": polygon.tolist()})
            if light_state == "RED":
                return light_state, metadata
            if light_state != "UNKNOWN" and first_detected_state is None:
                first_detected_state = (light_state, metadata)

        return first_detected_state if first_detected_state is not None else ("UNKNOWN", {})

    def _update_track_history(self, tracks: Iterable[TrackDetection]) -> None:
        for track in tracks:
            if track.track_id >= 0:
                self.centroid_history[track.track_id].append(track.centroid)

    def _wrong_side_violation(self, track: TrackDetection) -> tuple[bool, dict[str, Any]]:
        history = self.centroid_history.get(track.track_id)
        if history is None or len(history) < history.maxlen:
            return False, {}

        start = np.asarray(history[0], dtype=np.float32)
        end = np.asarray(history[-1], dtype=np.float32)
        movement = end - start
        displacement = float(np.linalg.norm(movement))
        delta_x = float(movement[0])
        delta_y = float(movement[1])

        legal_vector_points_down = float(self.allowed_lane_vector[1]) > 0.0
        moving_strictly_up = delta_y < -self.wrong_side_min_displacement
        if not legal_vector_points_down or not moving_strictly_up:
            return False, {}

        return True, {
            "delta_x_px": round(delta_x, 2),
            "delta_y_px": round(delta_y, 2),
            "displacement_px": round(displacement, 2),
            "history_frames": len(history),
            "legal_lane_vector": [round(float(value), 4) for value in self.allowed_lane_vector],
        }

    def _illegal_parking_violation(
        self,
        track: TrackDetection,
        no_parking_polygon: np.ndarray,
    ) -> tuple[bool, dict[str, Any]]:
        if track.track_id < 0 or not self._point_inside_polygon(track.centroid, no_parking_polygon):
            self.parking_state.pop(track.track_id, None)
            return False, {}

        history = self.parking_state[track.track_id]
        history.append(track.centroid)

        if len(history) < self.parking_stationary_frames:
            return False, {
                "roi": "no_parking",
                "stationary_frames": len(history),
                "required_frames": self.parking_stationary_frames,
            }

        start = np.asarray(history[0], dtype=np.float32)
        end = np.asarray(history[-1], dtype=np.float32)
        movement = float(np.linalg.norm(end - start))
        average_speed = movement / max(len(history) - 1, 1)

        return movement < self.parking_stationary_pixels, {
            "roi": "no_parking",
            "stationary_frames": len(history),
            "movement_px": round(movement, 2),
            "average_speed_px_per_frame": round(average_speed, 4),
            "threshold_px": self.parking_stationary_pixels,
        }

    def _append_violation(
        self,
        violations: list[ViolationRecord],
        seen: set[tuple[int, str]],
        track_id: int,
        violation_type: str,
        bbox: BBox,
        confidence: float | None = None,
        related_track_ids: Sequence[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        key = (track_id, violation_type)
        if key in seen:
            return

        seen.add(key)
        violations.append(
            ViolationRecord(
                track_id=track_id,
                violation_type=violation_type,
                bbox=bbox,
                confidence=confidence,
                related_track_ids=list(related_track_ids or []),
                metadata=metadata or {},
            )
        )

    def _draw_label(self, frame: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.48
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        y = max(y, text_size[1] + 6)
        cv2.rectangle(
            frame,
            (x, y - text_size[1] - 6),
            (x + text_size[0] + 6, y + baseline),
            color,
            -1,
        )
        cv2.putText(frame, text, (x + 3, y - 3), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_triple_riding_overlay(
        self,
        frame: np.ndarray,
        triple_violation_moto_bboxes: set[BBox],
        triple_rider_centroids: dict[BBox, list[tuple[float, float]]],
    ) -> None:
        """
        Draw the Triple Riding violation overlay without requiring pose keypoints.

        For each violating motorcycle:
          - Draw a thick RED bounding box (thickness=4)
          - Draw a neon ORANGE dot at each unique rider centroid (radius=6)
          - Print the rider count as a label inside the box
        """
        _VIOLATION_RED  = (0, 0, 255)
        _CENTROID_COLOR = (0, 165, 255)   # BGR orange – distinct from the box

        for moto_bbox in triple_violation_moto_bboxes:
            x1, y1, x2, y2 = moto_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), _VIOLATION_RED, 4)

            centroids = triple_rider_centroids.get(moto_bbox, [])
            count = len(centroids)
            for cx, cy in centroids:
                cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, _CENTROID_COLOR, -1, cv2.LINE_AA)
                cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, _VIOLATION_RED,  1, cv2.LINE_AA)

            if count > 0:
                label = f"TRIPLE: {count} riders"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(label, font, 0.55, 2)
                lx, ly = x1 + 4, y1 + th + 6
                cv2.rectangle(frame, (lx - 2, ly - th - 4), (lx + tw + 2, ly + 2), _VIOLATION_RED, -1)
                cv2.putText(frame, label, (lx, ly), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    def _annotate_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackDetection],
        violations: Sequence[ViolationRecord],
        stop_line_polygon: np.ndarray,
        no_parking_polygon: np.ndarray,
        light_state: str,
        triple_violation_moto_bboxes: set[BBox] | None = None,
        triple_rider_centroids: dict[BBox, list[tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        annotated = frame.copy()
        violation_types_by_track: dict[int, list[str]] = defaultdict(list)

        for violation in violations:
            violation_types_by_track[violation.track_id].append(violation.violation_type)
            for related_track_id in violation.related_track_ids:
                violation_types_by_track[related_track_id].append(violation.violation_type)

        cv2.polylines(annotated, [stop_line_polygon], True, self.STOP_LINE_COLOR, 2)
        cv2.polylines(annotated, [no_parking_polygon], True, self.NO_PARKING_COLOR, 2)

        for track in tracks:
            track_violations = violation_types_by_track.get(track.track_id, [])
            color = self.DEFAULT_BOX_COLOR
            if track_violations:
                color = self.VIOLATION_COLORS.get(track_violations[0], color)

            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            track_id_text = "NA" if track.track_id < 0 else str(track.track_id)
            label = f"{track.class_name} ID:{track_id_text}"
            if track_violations:
                label = f"{label} | {track_violations[0]}"
            self._draw_label(annotated, label, (x1, y1 - 6), color)

        signal_color = (0, 0, 255) if light_state == "RED" else (0, 180, 0)
        self._draw_label(annotated, f"Traffic Light: {light_state}", (12, 28), signal_color)

        # Draw geometric Triple Riding overlay on top of all other annotations
        self._draw_triple_riding_overlay(
            annotated,
            triple_violation_moto_bboxes or set(),
            triple_rider_centroids or {},
        )
        return annotated

    def process_frame(
        self,
        frame: np.ndarray,
        debug_mode: bool = False,
        stop_line_polygon_override: list[tuple[float, float]] | None = None,
        no_parking_polygon_override: list[tuple[float, float]] | None = None,
        manual_traffic_light: str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
        """
        Process one BGR frame and return an annotated frame, active violations, and
        an optional debug_payload.

        Returns
        -------
        annotated_frame : np.ndarray
            BGR frame with bounding boxes and labels drawn.
        result : dict[str, Any]
            Violations, tracked object counts, and frame metadata.
        debug_payload : dict[str, Any]
            Populated only when *debug_mode* is True; empty dict otherwise.
            Keys: ``raw_tracks``, ``helmet_crops``, ``seatbelt_crops``,
            ``ocr_variant_crops``, ``tracking_state``.
        """
        self._validate_frame(frame)
        self.frame_index += 1

        processed_frame = preprocess_frame(frame, mode=self.preprocessing_mode)
        track_kwargs: dict[str, Any] = {
            "source": processed_frame,
            "persist": True,
            "tracker": self.tracker_config,
            "conf": self.primary_conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": self.image_size,
            "verbose": False,
        }
        if self.device is not None:
            track_kwargs["device"] = self.device

        results = self.primary_model.track(**track_kwargs)
        result = results[0] if results else None
        all_tracks = self._extract_tracks(result, processed_frame.shape) if result is not None else []

        persons = self._filter_tracks(all_tracks, self.PERSON_LABELS)
        motorcycles = self._filter_tracks(all_tracks, self.MOTORCYCLE_LABELS)
        cars = self._filter_tracks(all_tracks, self.CAR_LABELS)
        monitored_tracks = persons + motorcycles + cars
        vehicle_tracks = motorcycles + cars

        # ------------------------------------------------------------------
        # Debug payload skeleton – populated below only when debug_mode=True
        # ------------------------------------------------------------------
        debug_payload: dict[str, Any] = {}
        if debug_mode:
            debug_payload["raw_tracks"] = [
                {
                    "track_id": t.track_id,
                    "class_name": t.class_name,
                    "confidence": round(t.confidence, 4),
                    "bbox": list(t.bbox),
                    "centroid": [round(t.centroid[0], 1), round(t.centroid[1], 1)],
                }
                for t in all_tracks
            ]
            debug_payload["helmet_crops"] = []   # filled in rider loop below
            debug_payload["seatbelt_crops"] = [] # filled in car loop below
            debug_payload["ocr_variant_crops"] = []  # filled after violations
            debug_payload["tracking_state"] = []  # filled after violations

        stop_line_polygon = self._resolve_polygon(
            stop_line_polygon_override if stop_line_polygon_override is not None else self.stop_line_polygon,
            processed_frame.shape,
            self._default_stop_line_polygon(processed_frame.shape),
        )
        no_parking_polygon = self._resolve_polygon(
            no_parking_polygon_override if no_parking_polygon_override is not None else self.no_parking_polygon,
            processed_frame.shape,
            self._default_no_parking_polygon(processed_frame.shape),
        )

        self._update_track_history(vehicle_tracks)
        
        if manual_traffic_light and manual_traffic_light != "Auto-Detect":
            light_state = manual_traffic_light
            light_metadata = {}
        else:
            light_state, light_metadata = self._detect_traffic_light_state(
                processed_frame,
                all_tracks,
                processed_frame.shape,
            )
        red_light_active = light_state in ["RED", "YELLOW"]

        violations: list[ViolationRecord] = []
        seen_violations: set[tuple[int, str]] = set()

        # ------------------------------------------------------------------
        # Triple Riding – dual-method: geometric centroid clustering
        #                              + helmet-count cross-reference
        # ------------------------------------------------------------------
        riders_by_motorcycle: dict[int, list[TrackDetection]] = {}
        # Maps motorcycle bbox -> unique rider centroids (for overlay drawing)
        triple_rider_centroids: dict[BBox, list[tuple[float, float]]] = {}

        for motorcycle in motorcycles:
            if not persons:
                # No persons detected – skip, nothing can trigger triple riding
                riders_by_motorcycle[motorcycle.track_id] = []
                continue

            # ---- Method A: geometric centroid clustering ------------------
            geo_violation, geo_count, unique_centroids = self._triple_riding_geometric(
                motorcycle, persons
            )

            # ---- Method B: helmet-count cross-reference ------------------
            helm_violation, helm_head_count = self._triple_riding_helmet_count(
                processed_frame, motorcycle
            )

            is_triple = geo_violation or helm_violation

            # Populate riders_by_motorcycle (for helmet loop below) using
            # the existing bbox-overlap heuristic so track_ids are available.
            rider_candidates = [
                p for p in persons if self._is_person_on_motorcycle(p, motorcycle)
            ]
            distinct_riders = self._nms_person_tracks(rider_candidates, iou_threshold=0.40)
            riders_by_motorcycle[motorcycle.track_id] = distinct_riders

            if is_triple:
                triple_rider_centroids[motorcycle.bbox] = unique_centroids
                self._append_violation(
                    violations,
                    seen_violations,
                    motorcycle.track_id,
                    "Triple Riding Violation",
                    motorcycle.bbox,
                    confidence=motorcycle.confidence,
                    related_track_ids=[r.track_id for r in distinct_riders],
                    metadata={
                        "detection_method": "geometric+helmet",
                        "geo_unique_riders": geo_count,
                        "geo_triggered": geo_violation,
                        "helmet_head_count": helm_head_count,
                        "helmet_triggered": helm_violation,
                        "centroid_merge_px": self._CENTROID_MERGE_PX,
                        "moto_expand_px": self._MOTO_EXPAND_PX,
                    },
                )

            # ---- Rider / Pillion classification + dual helmet violations ------
            labeled_riders = self._classify_riders_by_position(distinct_riders, motorcycle)
            for role, rider in labeled_riders:
                no_helmet_confidence = self._detect_no_helmet(processed_frame, rider)
                if debug_mode:
                    head_crop = self._helmet_crop(processed_frame, rider.bbox)
                    debug_payload["helmet_crops"].append({
                        "rider_track_id": rider.track_id,
                        "motorcycle_track_id": motorcycle.track_id,
                        "role": role,
                        "confidence": no_helmet_confidence,
                        "crop": head_crop,
                    })
                if no_helmet_confidence is not None:
                    violation_label = f"{role}: No Helmet"
                    self._append_violation(
                        violations,
                        seen_violations,
                        rider.track_id,
                        violation_label,
                        rider.bbox,
                        confidence=no_helmet_confidence,
                        related_track_ids=[motorcycle.track_id],
                        metadata={
                            "role": role,
                            "motorcycle_track_id": motorcycle.track_id,
                            "vehicle_bbox": [int(value) for value in motorcycle.bbox],
                        },
                    )

        for car in cars:
            no_seatbelt_confidence = self._detect_no_seatbelt(processed_frame, car)
            if debug_mode:
                sb_crop = self._seatbelt_crop(processed_frame, car.bbox)
                debug_payload["seatbelt_crops"].append({
                    "car_track_id": car.track_id,
                    "confidence": no_seatbelt_confidence,
                    "crop": sb_crop,
                })
            if no_seatbelt_confidence is not None:
                self._append_violation(
                    violations,
                    seen_violations,
                    car.track_id,
                    "Seatbelt Violation",
                    car.bbox,
                    confidence=no_seatbelt_confidence,
                    metadata={"crop_region": "windshield/front-seat"},
                )

        for vehicle in vehicle_tracks:
            is_wrong_side, wrong_side_metadata = self._wrong_side_violation(vehicle)
            if is_wrong_side:
                self._append_violation(
                    violations,
                    seen_violations,
                    vehicle.track_id,
                    "Wrong-side Violation",
                    vehicle.bbox,
                    confidence=vehicle.confidence,
                    metadata=wrong_side_metadata,
                )

            bottom_center = self._bbox_bottom_center(vehicle.bbox)
            inside_stop_line = self._point_inside_polygon(bottom_center, stop_line_polygon)
            if red_light_active and inside_stop_line:
                self._append_violation(
                    violations,
                    seen_violations,
                    vehicle.track_id,
                    "Red-light / Stop-line Violation",
                    vehicle.bbox,
                    confidence=vehicle.confidence,
                    metadata={
                        "light_state": light_state,
                        "vehicle_bottom_center": [
                            round(float(bottom_center[0]), 2),
                            round(float(bottom_center[1]), 2),
                        ],
                        "stop_line_polygon": stop_line_polygon.tolist(),
                        "traffic_light": light_metadata,
                    },
                )

            is_illegally_parked, parking_metadata = self._illegal_parking_violation(vehicle, no_parking_polygon)
            if is_illegally_parked:
                self._append_violation(
                    violations,
                    seen_violations,
                    vehicle.track_id,
                    "Illegal Parking Violation",
                    vehicle.bbox,
                    confidence=vehicle.confidence,
                    metadata=parking_metadata,
                )

        # Collect violating motorcycle bboxes and their rider centroids
        # for the geometric overlay drawing.
        triple_violation_moto_bboxes: set[BBox] = {
            v.bbox for v in violations if v.violation_type == "Triple Riding Violation"
        }

        annotated_frame = self._annotate_frame(
            processed_frame,
            monitored_tracks,
            violations,
            stop_line_polygon,
            no_parking_polygon,
            light_state,
            triple_violation_moto_bboxes=triple_violation_moto_bboxes,
            triple_rider_centroids=triple_rider_centroids,
        )

        # Populate remaining debug sections now that all violation logic is done
        if debug_mode:
            # OCR variant crops: collect for the first vehicle track that has a valid crop
            from core.ocr_engine import _ocr_variants  # local import to avoid circular
            for vehicle in vehicle_tracks:
                vcrop = self._safe_crop(processed_frame, vehicle.bbox)
                if vcrop is not None and vcrop.size > 0:
                    try:
                        variants = _ocr_variants(vcrop)
                        variant_labels = ["upscaled", "grayscale", "denoised", "adaptive", "inverted_adaptive"]
                        debug_payload["ocr_variant_crops"].append({
                            "vehicle_track_id": vehicle.track_id,
                            "variants": [
                                {"label": lbl, "crop": var}
                                for lbl, var in zip(variant_labels, variants)
                            ],
                        })
                    except Exception:
                        pass

            # Spatial heuristic state per tracked vehicle
            for vehicle in vehicle_tracks:
                bottom_center = self._bbox_bottom_center(vehicle.bbox)
                stop_line_poly_list = stop_line_polygon.tolist()
                no_parking_poly_list = no_parking_polygon.tolist()
                inside_stop = self._point_inside_polygon(bottom_center, stop_line_polygon)
                inside_no_parking = self._point_inside_polygon(vehicle.centroid, no_parking_polygon)
                parking_hist = self.parking_state.get(vehicle.track_id)
                centroid_hist = self.centroid_history.get(vehicle.track_id)
                debug_payload["tracking_state"].append({
                    "track_id": vehicle.track_id,
                    "class_name": vehicle.class_name,
                    "bbox": list(vehicle.bbox),
                    "centroid": [round(vehicle.centroid[0], 1), round(vehicle.centroid[1], 1)],
                    "inside_stop_line_zone": bool(inside_stop),
                    "inside_no_parking_zone": bool(inside_no_parking),
                    "red_light_active": bool(red_light_active),
                    "parking_frames_accumulated": len(parking_hist) if parking_hist else 0,
                    "parking_frames_required": self.parking_stationary_frames,
                    "centroid_history_frames": len(centroid_hist) if centroid_hist else 0,
                })

        # ------------------------------------------------------------------
        # Consolidated incident tracking
        # Update active_incidents with every violation seen this frame,
        # then flush incidents for vehicles that have left the frame.
        # ------------------------------------------------------------------
        active_track_ids: set[int] = {t.track_id for t in all_tracks if t.track_id >= 0}

        # Mark every active track as seen this frame
        for tid in active_track_ids:
            self.incident_last_seen[tid] = self.frame_index

        # Feed new violations into active_incidents
        for v in violations:
            primary_track_id = v.track_id
            # Use the vehicle (motorcycle/car) crop as evidence when available;
            # fall back to the violating object itself.
            v_crop = self._safe_crop(processed_frame, v.bbox)
            self._update_incident(primary_track_id, v, processed_frame, v_crop)

        # Flush departed incidents -> generate consolidated PDFs
        consolidated_challans = self._flush_departed_incidents(
            active_track_ids=active_track_ids,
            plate_text_by_track={},   # OCR lookup happens inside app.py; pass empty
        )

        result_payload = {
            "frame_index": self.frame_index,
            "red_light_active": red_light_active,
            "light_state": light_state,
            "tracked_objects": {
                "person": len(persons),
                "motorcycle": len(motorcycles),
                "car": len(cars),
            },
            "violations": [violation.to_dict() for violation in violations],
            "consolidated_challans": consolidated_challans,
        }
        return annotated_frame, result_payload, debug_payload
