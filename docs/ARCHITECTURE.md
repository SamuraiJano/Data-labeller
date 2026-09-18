# Architecture

The FastAPI process owns filesystem access, atomic persistence, validation, splitting, export, and optional segmentation providers. The React application owns interactive editing state and sends complete per-image revisions through an optimistic revision check.

```text
source folder -> project importer -> stable image records
                                  -> per-image annotation JSON
                                  -> full-resolution mask PNGs

fruit_box ------> YOLO26 detection + D-FINE detection
defect_mask ----> YOLO26 instance segmentation
foreground_mask -> tightly cropped RGBA synthetic asset

foreground assets + background assets -> seeded compositor with bounded, configurable overlap
                                      -> composite PNGs + transform manifest
                                      -> openable annotated project
```

Detection and segmentation classes are separate ordered lists. UUIDs are durable identity; `export_index` is recalculated as contiguous zero-based order. `foreground_mask` is classless. An explicit confirmed action is the only path from foreground geometry to a fruit box.

The split manifest is the single split authority. Exporters copy each oriented image into the same split for all formats. Classification uses the same assignments but places each image under `image_classification/<split>/<class>/`. Export paths are relative to their format roots and the top-level manifest contains no workstation-specific project path.

SAM is a replaceable annotation-provider boundary: official SAM 2.1 is the ready-to-use default and official SAM 3 remains an optional text-prompt provider. Models load lazily, current-image features are cached, and all failures remain local to proposal generation. Persistence, manual tools, validation, and export do not depend on SAM.
