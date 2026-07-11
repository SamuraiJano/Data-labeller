from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app import create_app
from .core import create_or_update_project
from .exporters import export_coco, export_yolo
from .previews import generate_previews
from .splitting import create_split_manifest, ensure_split_manifest
from .validation import write_validation_report


def _json_print(payload: object) -> None:
    print(json.dumps(payload, indent=2))


def add_common_project_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-dir", type=Path, default=Path("dataset") / "annotations", help="Directory containing project.json.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local citrus whole-fruit instance-segmentation labeler.")
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create or update an annotation project from a source image folder.")
    add_common_project_arg(create)
    create.add_argument("--source-dir", type=Path, required=True, help="Folder containing citrus canker, healthy and melanose subfolders.")
    create.add_argument("--name", default=None, help="Project name.")

    start = sub.add_parser("start", help="Start the local browser annotation app.")
    add_common_project_arg(start)
    start.add_argument("--source-dir", type=Path, default=None, help="Optional source folder to import if project.json does not exist.")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", type=int, default=8765)

    validate = sub.add_parser("validate", help="Validate project annotations and write reports.")
    add_common_project_arg(validate)
    validate.add_argument("--output-dir", type=Path, default=None)

    split = sub.add_parser("split", help="Create or reuse the deterministic split manifest.")
    add_common_project_arg(split)
    split.add_argument("--force", action="store_true", help="Regenerate the manifest.")
    split.add_argument("--seed", type=int, default=None)
    split.add_argument("--train", type=float, default=0.70)
    split.add_argument("--val", type=float, default=0.15)
    split.add_argument("--test", type=float, default=0.15)

    coco = sub.add_parser("export-coco", help="Export master COCO Instance Segmentation JSON.")
    add_common_project_arg(coco)
    coco.add_argument("--output-dir", type=Path, default=None)

    yolo = sub.add_parser("export-yolo", help="Export Ultralytics YOLO segmentation RGB and optional HSV datasets.")
    add_common_project_arg(yolo)
    yolo.add_argument("--output-dir", type=Path, default=Path("dataset") / "exports" / "yolo_seg")
    yolo.add_argument("--no-hsv", action="store_true")
    yolo.add_argument("--force-split", action="store_true")
    yolo.add_argument("--allow-invalid", action="store_true")

    previews = sub.add_parser("previews", help="Generate random overlay preview images.")
    add_common_project_arg(previews)
    previews.add_argument("--output-dir", type=Path, default=Path("dataset") / "exports" / "previews")
    previews.add_argument("--count", type=int, default=25)
    previews.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "create":
        _json_print(create_or_update_project(args.source_dir, args.project_dir, args.name))
    elif args.command == "start":
        import uvicorn

        app = create_app(args.project_dir, args.source_dir)
        print(f"Starting dataset labeler at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.command == "validate":
        _json_print(write_validation_report(args.project_dir, args.output_dir))
    elif args.command == "split":
        ratios = {"train": args.train, "val": args.val, "test": args.test}
        _json_print(create_split_manifest(args.project_dir, ratios, args.seed) if args.force else ensure_split_manifest(args.project_dir))
    elif args.command == "export-coco":
        _json_print(export_coco(args.project_dir, args.output_dir))
    elif args.command == "export-yolo":
        _json_print(
            export_yolo(
                args.project_dir,
                args.output_dir,
                include_hsv=not args.no_hsv,
                force_split=args.force_split,
                allow_invalid=args.allow_invalid,
            )
        )
    elif args.command == "previews":
        _json_print(generate_previews(args.project_dir, args.output_dir, args.count, args.seed))


if __name__ == "__main__":
    main()
