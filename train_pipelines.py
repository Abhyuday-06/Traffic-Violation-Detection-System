from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SIZE = 640
DEFAULT_EPOCHS = 15
DEFAULT_BATCH_SIZE = 16
NORMALIZED_CONFIG_DIR = PROJECT_ROOT / "configs" / "normalized_datasets"
TRAINING_PROJECT_DIR = PROJECT_ROOT / "runs" / "train"


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    model_name: str
    yaml_path: Path
    description: str


DATASETS: tuple[DatasetConfig, ...] = (
    DatasetConfig(
        key="helmet",
        model_name="helmet_detector",
        yaml_path=PROJECT_ROOT / "Helmet" / "data.yaml",
        description="Helmet, No-Helmet, and rider/person detector (original 3-class)",
    ),
    DatasetConfig(
        key="helmet_fixed",
        model_name="helmet_detector_2class",
        yaml_path=PROJECT_ROOT / "Helmet_fixed" / "data.yaml",
        description="Helmet / No-Helmet only — polygons converted to bbox, person class removed",
    ),
    DatasetConfig(
        key="helmet_v2",
        model_name="helmet_detector_v2",
        yaml_path=PROJECT_ROOT / "Helmet_v2" / "data.yaml",
        description="6.5k-image clean bbox dataset: Helmet / Motorcyclist / NoHelmet (CC BY 4.0)",
    ),
    DatasetConfig(
        key="seatbelt",
        model_name="seatbelt_detector",
        yaml_path=PROJECT_ROOT / "Seatbelt" / "data.yaml",
        description="Driver/passenger seatbelt compliance detector",
    ),
    DatasetConfig(
        key="numberplate2",
        model_name="license_plate_detector",
        yaml_path=PROJECT_ROOT / "NumberPlate2" / "data.yaml",
        description="Primary one-class license plate localizer",
    ),
    DatasetConfig(
        key="numberplate",
        model_name="license_plate_text_label_detector",
        yaml_path=PROJECT_ROOT / "NumberPlate" / "data.yaml",
        description="Secondary plate-labeled export with registration-text classes",
    ),
)


def _load_yaml(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid dataset YAML: {yaml_path}")
    return data


def _has_images(split_images_dir: Path) -> bool:
    if not split_images_dir.exists():
        return False

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return any(path.suffix.lower() in image_extensions for path in split_images_dir.iterdir())


def _normalize_dataset_yaml(dataset_yaml_path: str | Path) -> Path:
    """
    Create a runtime YAML with paths rooted at the dataset folder.

    Roboflow exports often contain entries like '../train/images'. In this workspace
    each export is nested under its own folder, so a normalized YAML avoids path
    resolution errors while leaving the original dataset files untouched.
    """
    source_yaml = Path(dataset_yaml_path).resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")

    dataset_dir = source_yaml.parent
    source_data = _load_yaml(source_yaml)

    train_images = dataset_dir / "train" / "images"
    valid_images = dataset_dir / "valid" / "images"
    test_images = dataset_dir / "test" / "images"

    if not _has_images(train_images):
        raise FileNotFoundError(f"Training images not found: {train_images}")
    if not _has_images(valid_images):
        raise FileNotFoundError(f"Validation images not found: {valid_images}")

    normalized_data: dict[str, Any] = {
        "path": str(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images" if test_images.exists() else "valid/images",
        "nc": source_data.get("nc"),
        "names": source_data.get("names"),
    }

    if normalized_data["nc"] is None or normalized_data["names"] is None:
        raise ValueError(f"Dataset YAML must define 'nc' and 'names': {source_yaml}")

    NORMALIZED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    normalized_yaml = NORMALIZED_CONFIG_DIR / f"{dataset_dir.name.lower()}_data.yaml"
    with normalized_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(normalized_data, file, sort_keys=False, allow_unicode=True)

    return normalized_yaml


def _select_validation_split(dataset_yaml_path: str | Path) -> str:
    dataset_dir = Path(dataset_yaml_path).resolve().parent
    test_images = dataset_dir / "test" / "images"
    return "test" if _has_images(test_images) else "val"


def train_model(
    dataset_yaml_path: str | Path,
    model_name: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | int | None = None,
) -> dict[str, str]:
    """
    Train a lightweight YOLOv8n detector using AdamW on a YOLO-format dataset.
    """
    normalized_yaml = _normalize_dataset_yaml(dataset_yaml_path)
    model = YOLO("yolov8s.pt")

    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    results = model.train(
        data=str(normalized_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=IMAGE_SIZE,
        optimizer="AdamW",
        project=str(TRAINING_PROJECT_DIR),
        name=model_name,
        save=True,
        save_period=5,
        pretrained=True,
        cache=False,
        workers=4,
        patience=10,
        amp=True,
        plots=True,
        exist_ok=True,
        device=device,
    )

    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    return {
        "model_name": model_name,
        "dataset_yaml": str(normalized_yaml),
        "save_dir": str(save_dir),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
    }


def _safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return 0.0
    return float(np.nanmean(array))


def _box_metric(metrics: Any, attribute_name: str, result_key: str) -> float:
    box_metrics = getattr(metrics, "box", None)
    if box_metrics is not None and hasattr(box_metrics, attribute_name):
        return _safe_float(getattr(box_metrics, attribute_name))

    results_dict = getattr(metrics, "results_dict", {}) or {}
    return _safe_float(results_dict.get(result_key, 0.0))


def _accuracy_from_confusion_matrix(metrics: Any) -> float | None:
    confusion_matrix = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(confusion_matrix, "matrix", None)
    if matrix is None:
        return None

    matrix_array = np.asarray(matrix, dtype=np.float64)
    if matrix_array.ndim != 2 or matrix_array.size == 0:
        return None

    class_count = len(getattr(metrics, "names", {}) or {})
    has_background_row = (
        class_count > 0
        and matrix_array.shape[0] == class_count + 1
        and matrix_array.shape[1] == class_count + 1
    )

    if has_background_row:
        object_matrix = matrix_array[:class_count, :class_count]
        false_positive_total = float(matrix_array[:class_count, class_count].sum())
        false_negative_total = float(matrix_array[class_count, :class_count].sum())
        denominator = float(object_matrix.sum()) + false_positive_total + false_negative_total
    else:
        object_matrix = matrix_array
        denominator = float(object_matrix.sum())

    if denominator <= 0:
        return 0.0

    return float(np.trace(object_matrix) / denominator)


def _detection_accuracy_from_precision_recall(precision: float, recall: float) -> float:
    """
    Estimate object-detection accuracy as TP / (TP + FP + FN).

    True negatives are not meaningful for dense object detection, so this uses the
    Jaccard-style detection accuracy derived from aggregate precision and recall.
    """
    denominator = precision + recall - (precision * recall)
    if denominator <= 0:
        return 0.0
    return float((precision * recall) / denominator)


def evaluate_model(
    weights_path: str | Path,
    dataset_yaml_path: str | Path,
    device: str | int | None = None,
) -> dict[str, float | str]:
    """
    Validate trained weights and report Accuracy, Precision, Recall, F1, and mAP.
    """
    weights = Path(weights_path).resolve()
    if not weights.exists():
        raise FileNotFoundError(f"Weights file not found: {weights}")

    normalized_yaml = _normalize_dataset_yaml(dataset_yaml_path)
    split = _select_validation_split(dataset_yaml_path)
    model = YOLO(str(weights))

    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    metrics = model.val(
        data=str(normalized_yaml),
        split=split,
        imgsz=IMAGE_SIZE,
        batch=DEFAULT_BATCH_SIZE,
        plots=True,
        device=device,
    )

    precision = _box_metric(metrics, "mp", "metrics/precision(B)")
    recall = _box_metric(metrics, "mr", "metrics/recall(B)")
    f1_score = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    map_50_95 = _box_metric(metrics, "map", "metrics/mAP50-95(B)")

    accuracy = _accuracy_from_confusion_matrix(metrics)
    if accuracy is None:
        accuracy = _detection_accuracy_from_precision_recall(precision, recall)

    evaluation = {
        "weights_path": str(weights),
        "dataset_yaml": str(normalized_yaml),
        "split": split,
        "Accuracy": round(float(accuracy), 6),
        "Precision": round(float(precision), 6),
        "Recall": round(float(recall), 6),
        "F1-Score": round(float(f1_score), 6),
        "mAP@50-95": round(float(map_50_95), 6),
    }

    print(json.dumps(evaluation, indent=2))
    return evaluation


def _selected_datasets(selected_keys: list[str] | None) -> list[DatasetConfig]:
    if not selected_keys:
        return list(DATASETS)

    dataset_by_key = {dataset.key: dataset for dataset in DATASETS}
    return [dataset_by_key[key] for key in selected_keys]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and validate YOLOv8 custom detectors for traffic violation analysis."
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (e.g. 0 or 'cpu'). Defaults to GPU 0 if available.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[dataset.key for dataset in DATASETS],
        help="Train only selected dataset keys. Defaults to all datasets sequentially.",
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip validation after training.",
    )

    args = parser.parse_args()
    selected_datasets = _selected_datasets(args.only)

    for dataset in selected_datasets:
        print(f"\n=== Training {dataset.model_name}: {dataset.description} ===")
        training_output = train_model(
            dataset_yaml_path=dataset.yaml_path,
            model_name=dataset.model_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )

        best_weights = Path(training_output["best_weights"])
        if args.no_eval:
            continue

        if best_weights.exists():
            print(f"\n=== Evaluating {dataset.model_name} on holdout split ===")
            evaluate_model(best_weights, dataset.yaml_path, device=args.device)
        else:
            print(f"Skipping evaluation because best weights were not found: {best_weights}")


if __name__ == "__main__":
    main()
