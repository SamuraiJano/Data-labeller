# Data Labeller

Local-first annotation and dataset-export application for the computer-vision project. It turns videos into full-resolution labeling frames, manages object-detection boxes, instance-segmentation masks, or image-level classification labels, protects review state, creates leakage-safe deterministic splits, and writes training-ready exports. It does not train or run production inference.

## Video-to-dataset workflow

From the start screen choose **New video dataset**, select any supported MP4, MOV, M4V, AVI, MKV, MPEG, or WebM files, then choose Detection, Segmentation, or Classification and enter the class names. The app extracts a JPEG at 0.0, 0.5, 1.0 seconds and so on while keeping the full decoded frame resolution and leaving the original videos untouched. Extraction runs in the background with progress and cancellation.

Extracted frames live inside the project. Every frame records its source-video fingerprint and target/actual timestamp. The automatic seed-42 split assigns complete videos to 70% training, 20% validation, and 10% test, so adjacent frames from one video cannot leak across splits. With fewer than three videos, the project remains usable and clearly identifies which independent split could not be created.

The labeling workspace is canvas-first: the image browser and annotation properties are overlay drawers, and the toolbar shows only the controls needed for the selected task. Detection uses box tools. Segmentation opens its mask controls automatically and offers one-click class-specific masks before painting, polygon editing, or optional SAM proposals. Classification presents large image-class buttons and supports number keys 1–9. Use **Done** for a completed detection/segmentation image, **No objects** for an intentional negative, or **Approve label** for classification, then open **Review & export** to validate and write the task-appropriate dataset.

## Annotation model

Every image and annotation has a stable UUID. Geometry is stored in full image pixels and masks remain full-resolution binary PNGs.

- `fruit_box`: `[x, y, width, height]` around one whole fruit; uses the configurable detection-class schema.
- `defect_mask`: one defect instance mask; uses the separate configurable segmentation-class schema and may reference a parent fruit UUID.
- `foreground_mask`: classless extraction mask used only for synthetic assets. The explicit derive action may create a `fruit_box`; the types are never silently converted or mixed.
- Classification labels live on the image record as one class UUID/name/export index; classification projects do not mix image labels with instance annotations.

Each annotation records class UUID/name/export index, image dimensions, source (`manual`, Grounding DINO, `SAM2`, `SAM3`, or import/migration), timestamps, prompts/provenance, review state, and optional parent UUID. Project state, per-image records, masks, revisions, undo history during the session, autosave, and recoverable backups protect work across restarts.

## Setup and run

Install Python 3.12 or newer and Node.js, then run this one command from the repository root:

```powershell
.\start.cmd
```

The launcher creates `.venv`, installs changed Python or frontend dependencies, builds changed frontend files, starts the backend, and opens `http://127.0.0.1:8765`. Later launches skip unchanged setup work. On macOS or Linux, use `python3 run.py` instead.

Optional settings can be supplied on the same command, for example `.\start.cmd --provider sam3 --port 9000`. Use `--no-browser` to prevent the browser from opening, or `--help` to see every option.

Create/rescan a project by choosing a source image folder and a separate project folder. Imports preserve relative source paths, compute checksums, read EXIF-correct dimensions, skip unreadable files with warnings, and keep stable IDs across rescans.

The workspace provides a searchable/status-filterable gallery, full-resolution canvas, zoom/pan/fit, mask opacity, add/erase brush, polygon fill, instance selection, multiple instances, numeric box move/resize, visibility/locking, parent assignment, undo/redo, autosave, exclusion reasons, review/approval, and help/shortcuts. Previous/next frame buttons remain visible in the header, and the canvas toolbar includes an enabled-state-aware **Undo mask edit** action plus zoom-out, percentage, zoom-in, and fit controls. The mouse wheel zooms around the pointer; Space-drag or Pan moves around an enlarged frame. Returning to an approved frame and changing its mask reopens it as a draft for review. The pencil action opens the source-pixel editor: zoom with the mouse wheel or dedicated controls, enable Pan to move around an enlarged image, and paint over unwanted GrabCut remnants to make those source pixels transparent. Individual strokes can be undone or the preview reset before applying the change. JPEG/BMP inputs are converted to PNG when transparency is first added. The same editor also provides rotate-left, rotate-right, and horizontal/vertical flips; boxes and masks are transformed with geometric edits, and the original source file is copied to the project's recoverable backups before any source edit. In the image drawer, use the row checkboxes, Shift-click ranges, or **Select filtered** to select many images, then use **No instances** to approve the whole selection as valid negatives in one conflict-checked save. Press **N** to mark the current image as having no instances, or use **Left Arrow** and **Right Arrow** for previous/next. Labelled actions delete the selected annotation, clear only its mask pixels, or clear every annotation on the current image after confirmation; saved masks move into recoverable backups. **Change folder** safely saves and returns to the project picker without deleting the current project.

## AI-assisted initial bounding boxes

Detection and mixed projects include an **AI boxes** button in the canvas toolbar. It opens an Apache-2.0 [Grounding DINO Tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) assistant that can search the current image using ordinary descriptions such as `orange`, `naartjie`, or `citrus fruit`. The model is open-vocabulary, so a citrus-specific trained checkpoint is not required to get useful starting boxes.

Click **Load detector** once, then **Find boxes**. Set **Maximum boxes** from 1 to 100 when an image may contain more than the default 25 objects. The first load downloads the model files from Hugging Face and caches them; inference thereafter runs locally and source images are not uploaded. Suggested boxes appear as dashed yellow previews. They are not annotations and are not saved until **Use** or **Use all boxes** is clicked. Every accepted box is an editable draft with the model, prompt, predicted label, and confidence recorded in its provenance. Resize or move it normally, delete false positives, and only then mark the image Done.

`start.cmd` installs the detector dependencies automatically. When it finds an NVIDIA GPU, it also replaces a CPU-only PyTorch package with the official CUDA build; the next detector load reports `cuda` in the AI-boxes panel. For a manual environment, install the project and CUDA PyTorch with `python -m pip install -e ".[detection]"` followed by `python -m pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu130`. `GROUNDING_DINO_DEVICE=auto` uses CUDA when PyTorch detects a compatible GPU and otherwise uses the CPU; set it to `cpu` to explicitly opt out. The integration uses Hugging Face's [official Transformers interface](https://huggingface.co/docs/transformers/model_doc/grounding-dino) and the [official Grounding DINO repository](https://github.com/IDEA-Research/GroundingDINO).

## SAM-assisted initial masks

Segmentation projects expose a prominent **SAM mask** action alongside the manual mask tools. Choose the instance class, add a positive point or tight box on the image, preview up to three cyan alternatives, and accept one as an editable draft. The include/exclude points, box, model score, and provider are retained as provenance. Finish the boundary with Add, Erase, or Polygon before marking the image done.

New segmentation projects start with these ordered defect classes: **Skin wrinkling**, **Dark marking**, **Dark scarring**, **Wind scarring**, **Light-brown discolouration**, **Rough skin**, and **Open skin**. Their zero-based export indices follow that order.

The normal launcher installs Meta's official SAM 2.1 package at the pinned revision and selects it by default. The repository-local `checkpoints/sam2.1_hiera_small.pt` is checksum-verified before loading, and `SAM2_DEVICE=auto` uses CUDA on a compatible workstation. Model loading is lazy, and the current image embedding is cached across prompt refinements. Manual masks remain fully usable if SAM cannot load.

```powershell
.\start.cmd
# Explicit alternatives for UI testing or an optional SAM 3 installation:
.\start.cmd --provider mock
.\start.cmd --provider sam3
```

SAM 3 remains an optional text-and-box prompting provider. It imports Meta's official `facebookresearch/sam3` package and uses `build_sam3_image_model` plus `Sam3Processor`; it does not substitute a similarly named third-party model.

## Defective-naartjie video harvester

Choose **Harvest defective fruit from video** from the home screen or Tools menu to build a focused dataset without labelling thousands of empty frames. Play a source video, pause when a defective naartjie is visible, and click the whole fruit. SAM 2 previews the fruit mask; positive/negative points, a manual box, or optional Grounding DINO fruit boxes can refine the seed. Accepting the seed creates a persistent track. **Save & Next 0.25s** writes the current whole-fruit RGBA cut-out and propagates every active track through the intervening native video frames before showing the next sample. Multiple fruit may be tracked together, and Auto-step repeats this operation at the model's processing speed.

Harvest sessions are resumable and never modify the videos. The browser remembers the latest active session folder, returns to the harvester on the next application launch, and restores the saved video, timestamp, masks, and active tracks automatically. **Start another harvest** clears that remembered pointer without deleting the previous session from disk. Masks that touch the source-frame boundary are saved normally; sessions created by older versions automatically recover retained masks that were skipped only for touching that boundary. Tiny, fragmented, or implausibly changing masks are still skipped and reported. Finishing creates `foregrounds`, `harvest-session.json`, `harvest-manifest.json`, and an unlabelled segmentation project. The generated project contains no automatic defect annotations: every transparent PNG is a whole naartjie ready for manual defect-mask labelling. All views from one track share a `fruit_id` so they remain in one dataset split.

```powershell
git clone https://github.com/facebookresearch/sam3 C:\src\sam3
git -C C:\src\sam3 checkout 46957e47805eaa273f4aa7bbbd25a88bca9108ce
python -m pip install -e C:\src\sam3
Copy-Item .env.example .env
```

Configure `SAM3_REPO_PATH`, `SAM3_DEVICE`, and either `SAM3_CHECKPOINT_PATH` or `SAM3_LOAD_FROM_HF=true`. No access token or user-specific path is committed. The official package currently requires Python 3.12+, PyTorch 2.7+, and a supported CUDA stack; the app reports missing packages, unsupported Torch/CUDA, checkpoint errors, gated-download failures, and OOM without disabling manual annotation. Text and normalized box prompts use the official image processor.

## Split and export

Default splitting is deterministic 70% train / 20% validation / 10% test with seed 42. Image-folder datasets are stratified independently by their top-level source-class folder. Explicit source-video, capture-group, or fruit IDs keep related images together; filenames ending in a fruit number and view number (for example `klas1_001_1.png`) also infer a physical-fruit group, so every view of one naartjie remains in a single split. The test membership can be frozen and is intended only for final evaluation after model training; tune with the training and validation sets. Each image receives one split reused by every export format. Generated project folders such as `thumbnails`, `masks`, and `annotations` are excluded from source-image rescans.

```powershell
python -m tools.dataset_labeler split --project-dir dataset\annotations --force --train 0.7 --val 0.2 --test 0.1 --seed 42
python -m tools.dataset_labeler validate --project-dir dataset\annotations
python -m tools.dataset_labeler export-dataset --project-dir dataset\annotations --output-dir dataset\exports
```

The result is `<project>-training-export/`:

```text
export_manifest.json
class_schema.json
yolo26_detection/{data.yaml,images/{train,val,test},labels/{train,val,test}}
yolo26_segmentation/{data.yaml,images/{train,val,test},labels/{train,val,test}}
dfine_detection/{dataset_info.yaml,custom_detection.yml,images/{train,val,test},annotations/instances_{train,val,test}.json}
synthetic_assets/{foregrounds,backgrounds,assets_manifest.json}
image_classification/{dataset.yaml,train/<class>,val/<class>,test/<class>}
```

YOLO detection rows are zero-based normalized `class x_center y_center width height`. YOLO segmentation writes one zero-based normalized polygon row per disconnected defect component, with at least three vertices. Empty/missing instance labels are valid negatives. Classification exports the standard ImageFolder layout consumed directly by Ultralytics classification and `torchvision.datasets.ImageFolder`; every included image must have exactly one configured class. Class folders receive stable zero-padded index prefixes (for example `000-good`) so alphabetical ImageFolder discovery preserves the configured export indices; `dataset.yaml` retains the clean names and folder mapping. D-FINE receives COCO pixel `xywh`, positive `area`, `iscrowd: 0`, and contiguous zero-based category IDs with `remap_mscoco_category: false`, matching the current official loader. Foregrounds are tightly cropped RGBA files named by annotation UUID; backgrounds are separate.

Typical downstream training entry points are:

```powershell
yolo segment train model=yolo26n-seg.pt data="PATH\yolo26_segmentation\data.yaml"
yolo classify train model=yolo26n-cls.pt data="PATH\image_classification"
```

`export_manifest.json` contains only portable relative paths plus the seed/ratios/split counts, class UUID/name/index mappings, source-image records, annotation UUIDs, checksums, counts, warnings, formats, and export hash.

## Synthetic-data generation

The main menu includes a **Generate composite images** section. Choose a foregrounds folder, a backgrounds folder, an output parent folder, the exact total number of images, and a maximum of 1–300 objects per image. Foreground source files are deterministically partitioned 80%/20% into disjoint training and validation pools before compositing, so a foreground used in training is never used in validation. That maximum is the number pasted into one composite, not the number of foreground source files; each split's shuffled deck automatically uses every foreground assigned to that split before reusing one. Each image receives a seeded random object count from zero through that maximum, and every generation includes at least one untouched background image as a valid negative. By default, each selected foreground is resized to a random 10–17% of its original dimensions, rotated between 0° and 360°, and placed at a random x/y position.

Controlled occlusion is enabled by default: 30% of eligible foregrounds are deliberately positioned over another fruit, while no fruit may share more than 20% of its alpha-mask pixels. Both percentages are configurable, and setting the overlap chance to zero restores non-overlapping placement. Lower objects' masks and training boxes are clipped to their visible pixels; the manifest also records their full pre-occlusion boxes, overlap fractions, occlusion fractions, and visible fractions.

The requested image total is exact. Every background is used, and each split's shuffled foreground deck uses every assigned foreground before any is reused. The object count is an upper bound: when a composite reaches physical capacity within the configured overlap limit, the app keeps that valid image and defers the next foreground to a later image in the same split. Capacity-limited images, planned-versus-actual counts, foreground split membership, and the reason are recorded in the manifest and summarized as a warning. Generation fails only when the complete run cannot place every source foreground at least once. The seed makes the foreground partition, object-count plan, and placements reproducible.

Generation runs as a background job. The main menu shows a live percentage bar, completed/total composite count, and the current stage while masks, boxes, splits, and YOLO files are being written.

The result is `<output>/synthetic-data/`, containing `synthetic_generation_manifest.json`, an immediately openable `project/`, and one copy of every composite under `yolo_detection/images/{train,val}`. There is no duplicate top-level `images/` tree: the project reads its source images directly from the YOLO folders. `yolo_detection/` is directly trainable and includes matching label files in normalized `class x_center y_center width height` format plus `data.yaml`. Each training box is calculated tightly from the visible transformed alpha mask, while the manifest retains the full pre-occlusion box. Re-running into the same location preserves the previous output as a timestamped sibling folder.

## Repository boundary

- Data Labeller: images, annotations, foreground/background assets, synthetic composite generation, splits, dataset export.
- Model Trainer: export validation, YOLO26-N/D-FINE-N training, runs, metrics, checkpoints.
- Inference Runner: prediction, cameras/video, tracking/aggregation, deployment and Raspberry Pi execution.

ByteTrack and BoT-SORT are runtime tracker configurations; they are not trained here.

## Synthetic paired blur datasets

Open the **Synthetic Blur Dataset** tab to preview and generate aligned `blurred input -> sharp target` pairs. The generator applies the same EXIF correction, centre crop/resize or padding to the target first; only the blurred member then receives a sampled motion-path PSF, small exposure/brightness/contrast changes, and low sensor-noise approximation. Horizontal conveyor, slight diagonal, curved/tangential, and smoothly blended spatially varying motion are independently configurable, with separate moderate and severe ranges. Gaussian blur is not used as the motion model.

Sources are grouped before variants are created. Exact hashes, matching average perceptual hashes, and optional `capture_group`, `fruit_id`, `tree_id`, or `session_id` values from `source_groups.csv`, `source_groups.json`, or an existing `manifests/split_manifest.json` remain in one split. Review the preview split counts and grouping metadata; perceptual grouping is a safeguard, not a replacement for correct capture metadata.

```powershell
python -m tools.dataset_labeler generate-deblur `
  --source-dir "PATH" --output-dir "PATH\deblur_dataset" `
  --size 256 --variants 4 --seed 42
python -m tools.dataset_labeler validate-deblur --dataset-dir "PATH\deblur_dataset"
# An interrupted run can be continued only when its source/config fingerprint matches:
python -m tools.dataset_labeler generate-deblur --source-dir "PATH" --output-dir "PATH\deblur_dataset" --resume
```

```text
deblur_dataset/
  train/{blur,sharp}/
  val/{blur,sharp}/
  test/{blur,sharp}/
  manifests/{pairs.csv,pairs.json,generation_config.json}
```

Writes are atomic, existing non-resume outputs are rejected, and resume validates recorded hashes, RGB mode, and dimensions before continuing. Manifests retain the source and output hashes, split/group identities, pair seed, trajectory/kernel settings, photometric/noise settings, and sharp/blurred Laplacian variance. The UI provides deterministic preview/difference images, storage and split estimates, live progress, cancellation, resume, per-file failure reporting, and post-generation validation.

Synthetic blur is an initial training method. Final conclusions require real conveyor-blur validation: the synthetic-to-real domain gap can be substantial, and sharper-looking output does not prove improved defect detection. Compare deblurred and original crops in the Model Trainer safety evaluation before deployment.

The paired-data generator is project code and vendors no third-party implementation. The downstream training design is informed by [Ghost-DeblurGAN](https://github.com/York-SDCNLab/Ghost-DeblurGAN) (MIT) and [DeblurGAN-v2](https://github.com/VITA-Group/DeblurGANv2); their code is not cloned at application runtime. See the Model Trainer repository for architecture attribution, licences, and the AGG paper-reproduction disclaimer.

Run regression checks with:

```powershell
python -m pytest
cd frontend
npm test
npm run build
```
