from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app import create_app
from .core import create_or_update_project
from .deblur_dataset import BLUR_TYPES, DeblurGenerationConfig, generate_deblur_dataset, validate_deblur_dataset
from .exporters import export_coco, export_dataset
from .previews import generate_previews
from .splitting import create_split_manifest, ensure_split_manifest
from .validation import write_validation_report


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def add_common_project_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", type=Path, default=Path("dataset") / "annotations", help="Data Labeller project root.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local image annotation and portable training-data export app.")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create or rescan a project from one image folder.")
    add_common_project_arg(create)
    create.add_argument("--source-dir", type=Path, required=True)
    create.add_argument("--name", default=None)

    start = sub.add_parser("start", help="Start the local FastAPI and built React application.")
    add_common_project_arg(start)
    start.add_argument("--source-dir", type=Path, default=None)
    start.add_argument("--provider", choices=("mock", "sam2", "sam3"), default=None)
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8765)

    validate = sub.add_parser("validate", help="Validate images, masks, metadata and split leakage.")
    add_common_project_arg(validate)
    validate.add_argument("--output-dir", type=Path, default=None)

    split = sub.add_parser("split", help="Create or reuse the deterministic image/group split manifest.")
    add_common_project_arg(split)
    split.add_argument("--force", action="store_true")
    split.add_argument("--pilot-mode", action="store_true", help="Compatibility flag; image-level splitting is supported by default.")
    split.add_argument("--freeze-test", action="store_true")
    split.add_argument("--confirm-frozen-change", action="store_true")
    split.add_argument("--dataset-version", default=None)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--val", type=float, default=0.20)
    split.add_argument("--test", type=float, default=0.10)

    coco = sub.add_parser("export-coco", help="Optional COCO JSON derived from source masks.")
    add_common_project_arg(coco)
    coco.add_argument("--output-dir", type=Path, default=None)

    dataset = sub.add_parser("export-dataset", aliases=["export-yolo"], help="Export task-specific detection, segmentation, classification, and synthetic training data.")
    add_common_project_arg(dataset)
    dataset.add_argument("--output-dir", type=Path, default=Path("dataset") / "exports")
    dataset.add_argument("--minimum-iou", type=float, default=0.98)
    dataset.add_argument("--simplification-pixels", type=float, default=0.5)
    dataset.add_argument("--precision", type=int, default=6)
    dataset.add_argument("--force-split", action="store_true")
    dataset.add_argument("--block-warnings", action="store_true")

    previews = sub.add_parser("previews", help="Generate random mask overlay previews.")
    add_common_project_arg(previews)
    previews.add_argument("--output-dir", type=Path, default=Path("dataset") / "exports" / "previews")
    previews.add_argument("--count", type=int, default=25)
    previews.add_argument("--seed", type=int, default=42)

    deblur = sub.add_parser("generate-deblur", help="Generate an aligned sharp/motion-blurred paired dataset.")
    deblur.add_argument("--source-dir", type=Path, required=True)
    deblur.add_argument("--output-dir", type=Path, required=True)
    deblur.add_argument("--size", type=int, default=256)
    deblur.add_argument("--variants", type=int, default=4)
    deblur.add_argument("--seed", type=int, default=42)
    deblur.add_argument("--train", type=float, default=80.0, help="Train percentage.")
    deblur.add_argument("--val", type=float, default=10.0, help="Validation percentage.")
    deblur.add_argument("--test", type=float, default=10.0, help="Test percentage.")
    deblur.add_argument("--disable-blur", action="append", choices=BLUR_TYPES, default=[])
    deblur.add_argument("--blur-probability", action="append", default=[], metavar="TYPE=WEIGHT")
    deblur.add_argument("--moderate-length", nargs=2, type=int, default=(5, 13), metavar=("MIN", "MAX"))
    deblur.add_argument("--severe-length", nargs=2, type=int, default=(15, 31), metavar=("MIN", "MAX"))
    deblur.add_argument("--moderate-probability", type=float, default=0.65)
    deblur.add_argument("--resize-mode", choices=("cover", "contain"), default="cover")
    deblur.add_argument("--resume", action="store_true")

    validate_deblur = sub.add_parser("validate-deblur", help="Validate a paired deblurring dataset and leakage constraints.")
    validate_deblur.add_argument("--dataset-dir", type=Path, required=True)
    return parser.parse_args()


def _deblur_config(args: argparse.Namespace) -> DeblurGenerationConfig:
    probabilities = dict(DeblurGenerationConfig().blur_probabilities)
    for item in args.blur_probability:
        try:
            name, value = item.split("=", 1)
            if name not in BLUR_TYPES:
                raise ValueError
            probabilities[name] = float(value)
        except ValueError as exc:
            raise SystemExit(f"--blur-probability must be TYPE=WEIGHT where TYPE is one of {', '.join(BLUR_TYPES)}") from exc
    enabled = tuple(name for name in BLUR_TYPES if name not in args.disable_blur)
    return DeblurGenerationConfig.from_mapping({
        "size": args.size, "variants": args.variants, "seed": args.seed,
        "ratios": {"train": args.train / 100.0, "val": args.val / 100.0, "test": args.test / 100.0},
        "enabled_blur_types": enabled, "blur_probabilities": probabilities,
        "moderate_probability": args.moderate_probability, "moderate": {"path_length": args.moderate_length},
        "severe": {"path_length": args.severe_length}, "resize_mode": args.resize_mode,
    })


def main() -> None:
    args = parse_args()
    if args.command == "create":
        _json_print(create_or_update_project(args.source_dir, args.project_dir, args.name))
    elif args.command == "start":
        import uvicorn

        app = create_app(args.project_dir, args.source_dir, args.provider)
        print(f"Starting Data Labeller at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "validate":
        _json_print(write_validation_report(args.project_dir, args.output_dir))
    elif args.command == "split":
        if not args.force:
            _json_print(ensure_split_manifest(args.project_dir))
        else:
            _json_print(
                create_split_manifest(
                    args.project_dir,
                    {"train": args.train, "val": args.val, "test": args.test},
                    args.seed,
                    pilot_mode=args.pilot_mode,
                    freeze_test=args.freeze_test,
                    confirm_frozen_change=args.confirm_frozen_change,
                    dataset_version=args.dataset_version,
                )
            )
    elif args.command == "export-coco":
        _json_print(export_coco(args.project_dir, args.output_dir))
    elif args.command in {"export-dataset", "export-yolo"}:
        _json_print(
            export_dataset(
                args.project_dir,
                args.output_dir,
                minimum_iou=args.minimum_iou,
                simplification_pixels=args.simplification_pixels,
                precision=args.precision,
                force_split=args.force_split,
                allow_warnings=not args.block_warnings,
            )
        )
    elif args.command == "previews":
        _json_print(generate_previews(args.project_dir, args.output_dir, args.count, args.seed))
    elif args.command == "generate-deblur":
        _json_print(generate_deblur_dataset(args.source_dir, args.output_dir, _deblur_config(args), resume=args.resume))
    elif args.command == "validate-deblur":
        report = validate_deblur_dataset(args.dataset_dir)
        _json_print(report)
        if not report["valid"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
