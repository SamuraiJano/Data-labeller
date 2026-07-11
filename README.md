# Skripsie Segmentation Labeler

Local annotation tool for whole-fruit citrus instance segmentation. The current schema is:

```text
0: healthy
1: citrus_canker
2: melanose
```

Each instance is one complete visible citrus fruit. Because the selected definition is whole-fruit segmentation, images in `healthy/` also need fruit masks. Do not use lesion-only masks in the same project.

## Windows Setup

```powershell
cd "C:\Users\USER\Desktop\1Skripsie\VS Code\Segmentation labeler"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Install and build the React web app:

```powershell
cd frontend
npm install
npm run build
cd ..
```

If this Windows/npm environment reports `UNABLE_TO_VERIFY_LEAF_SIGNATURE`, retry the install command only with:

```powershell
npm install --strict-ssl=false
```

## Start The App

```powershell
python -m tools.dataset_labeler start --project-dir dataset\annotations
```

Open:

```text
http://127.0.0.1:8765
```

The Python server serves the built React app from `frontend\dist` and keeps the annotation, validation, split and export rules in the backend.

For frontend-only development, run the API and Vite dev server in two terminals:

```powershell
python -m tools.dataset_labeler start --project-dir dataset\annotations
cd frontend
npm run dev
```

Then open:

```text
http://127.0.0.1:5174
```

In the app, select the folder containing:

```text
citrus canker\
healthy\
melanose\
```

Then select or create the annotation project folder, normally `dataset\annotations`.

## Commands

Create or update a project from the terminal:

```powershell
python -m tools.dataset_labeler create --source-dir "D:\path\to\raw_citrus_images" --project-dir dataset\annotations
```

Validate annotations:

```powershell
python -m tools.dataset_labeler validate --project-dir dataset\annotations --output-dir dataset\exports\reports
```

Create or freeze the split manifest:

```powershell
python -m tools.dataset_labeler split --project-dir dataset\annotations --force --seed 42
```

Export master COCO:

```powershell
python -m tools.dataset_labeler export-coco --project-dir dataset\annotations --output-dir dataset\annotations
```

Export Ultralytics YOLO segmentation plus HSV:

```powershell
python -m tools.dataset_labeler export-yolo --project-dir dataset\annotations --output-dir dataset\exports\yolo_seg
```

Generate overlay previews:

```powershell
python -m tools.dataset_labeler previews --project-dir dataset\annotations --output-dir dataset\exports\previews --count 25
```

Run tests:

```powershell
python -m pytest
```

## Annotation Rules

Draw one polygon around each complete citrus fruit. Follow the fruit boundary as closely as the image quality allows, and keep vertices on the actual visible edge rather than on the background.

For partially occluded fruit, outline only the visible fruit surface. Do not invent hidden boundaries behind leaves, other fruit, labels or equipment.

If a fruit is too uncertain to assign to the source-folder class, mark the image as `excluded` and write an exclusion reason. Tiny ambiguous fragments should normally be excluded rather than guessed.

Healthy images are not negative images in this schema. They require `healthy` whole-fruit masks.

Autosave runs after each meaningful edit. The app writes `project.json` atomically and keeps timestamped backups under `dataset/annotations/backups/`.

Use `reviewed` only after checking that all visible fruits in the image have correct whole-fruit masks and class IDs.

## Formats

`project.json` is the editable project state. It stores original-image pixel coordinates with full precision, image dimensions, SHA-256 hashes, source folders, split state, review state and exclusion reasons.

`coco_instances.json` is the master interchange export. It preserves image metadata, categories, multiple instances, polygons, bounding boxes, area and `iscrowd`.

The YOLO export writes one `.txt` file per image. Each instance row is:

```text
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

Coordinates are normalized to `0..1`. Multiple fruit instances become multiple rows. Images without annotations can only be exported with `--allow-invalid`, and produce empty label files.

## Splitting

The split defaults are:

```text
train: 70%
val: 15%
test: 15%
seed: 42
```

Splits are deterministic. The splitter groups exact SHA-256 duplicates, inferred capture groups and explicit `capture_group`, `tree_id`, `fruit_id` or `session_id` metadata so related photographs do not leak across train, validation and test. Imported images also store a lightweight perceptual hash, and validation warns about perceptual duplicate candidates for manual review.

The split manifest is saved under:

```text
dataset\annotations\manifests\split_manifest.json
dataset\annotations\manifests\split_manifest.csv
```

It is reused unless you explicitly run `split --force`.

## RGB And HSV

Annotate the original RGB images once. The HSV export is derived from the RGB YOLO export and reuses the exact same labels and split manifest.

HSV images are written as OpenCV raw HSV channels using:

```python
cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
```

This matches the nearby `New Model Trainer` convention. The stored H channel uses OpenCV's `0..179` range, and S/V use `0..255`. Labels are copied byte-for-byte.

## Training

The generated RGB dataset is consumable directly by Ultralytics:

```powershell
cd "C:\Users\USER\Desktop\1Skripsie\VS Code\New Model Trainer\model_training"
python train_rgb.py --data "C:\Users\USER\Desktop\1Skripsie\VS Code\Segmentation labeler\dataset\exports\yolo_seg\data.yaml" --model yolo26n-seg.pt --epochs 100 --imgsz 640
```

Train HSV with the HSV `data.yaml`:

```powershell
python train_hsv.py --data "C:\Users\USER\Desktop\1Skripsie\VS Code\Segmentation labeler\dataset\exports\yolo_seg\hsv\data.yaml" --model yolo26n-seg.pt --epochs 100 --imgsz 640
```

The current C++ deployment code in `New Model Trainer` still has older display labels in places, so update deployment-facing names before presenting a canker/healthy/melanose model in that dashboard.
