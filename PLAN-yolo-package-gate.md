# Plan: YOLO Package-Presence Gate Pipeline

## Context
- The current five-label multi-label classifier (`damaged`, `plastic_wrap`, `sealed`, `open`, `non_package`) does not separate `non_package` reliably.
- Add a first-stage YOLO segmentation gate trained on Ultralytics' package segmentation dataset.
- If the gate finds a package, run the existing multi-label classifier; otherwise return `non_package`/stop early.
- Target deployment includes the Android smartphone app, so inference speed, model size, and memory impact are core requirements.

## Approach
- Create a new branch for the implementation.
- Add an Ultralytics YOLO segmentation training/export/evaluation path alongside the existing classifier pipeline.
- Use the newest available Ultralytics segmentation model family, starting with the smallest `n`/nano checkpoint for smartphone speed, and train/fine-tune it for a single `package` class.
- For gating, use detection confidence/output from the segmentation model; masks do not need to be parsed on-device unless later needed for debugging/visualization.
- Add a Python two-stage inference/benchmark script to measure:
  - YOLO gate latency
  - classifier latency
  - combined latency
  - skip rate
  - false-negative rate for package presence
- Export the YOLO gate to a mobile format and integrate it into the Android app before the existing classifier.
- Tune the YOLO confidence threshold for high recall, because gate false negatives suppress the downstream classifier entirely.

## Files to modify
- `pyproject.toml` / `uv.lock` — add Ultralytics/export dependencies if needed.
- `README.md` — document training, export, benchmark, and Android pipeline usage.
- New Python files likely under `pxmodel/`, for example:
  - `pxmodel/yolo_package_train.py`
  - `pxmodel/yolo_package_export.py`
  - `pxmodel/yolo_package_predict.py`
  - `pxmodel/pipeline_predict.py`
  - `pxmodel/pipeline_benchmark.py`
- Android files:
  - `android/app/src/main/java/com/pxmodel/classifier/Classifier.kt`
  - `android/app/src/main/java/com/pxmodel/classifier/MainActivity.kt`
  - possibly new helper classes for YOLO segmentation output parsing and preprocessing.
  - `android/app/src/main/assets/` for the exported YOLO gate model artifact.

## Reuse
- Existing classifier loading/inference logic:
  - `pxmodel/predict.py`
  - `pxmodel/export.py`
- Existing benchmark timing patterns:
  - `pxmodel/benchmark.py`
- Existing Android LiteRT/TFLite runtime setup:
  - `android/app/src/main/java/com/pxmodel/classifier/Classifier.kt`
- Existing app UI/result display flow:
  - `android/app/src/main/java/com/pxmodel/classifier/MainActivity.kt`

## Steps
- [ ] Create a new implementation branch, e.g. `feature/yolo-package-gate`.
- [ ] Confirm the newest available Ultralytics segmentation checkpoint name during implementation and select the smallest mobile-friendly variant first.
- [ ] Confirm the Ultralytics package segmentation dataset identifier/path and add scripts for downloading/preparing it.
- [ ] Add dependencies and scripts for YOLO package segmentation training.
- [ ] Train the smallest YOLO segmentation model first for one-class package presence.
- [ ] Evaluate package-presence recall/precision and tune confidence threshold for high recall.
- [ ] Export the YOLO gate to the mobile runtime format supported by the Android app.
- [ ] Add a Python two-stage inference path: YOLO gate first, existing multi-label classifier second.
- [ ] Add pipeline benchmarking to compare current single-classifier latency vs. YOLO-gated latency.
- [ ] Integrate the YOLO gate into Android before the existing classifier, using detection confidence for the gate decision.
- [ ] Show separate Android timings for gate, classifier, and total pipeline latency.
- [ ] Update docs with training/export/benchmark commands and threshold guidance.

## Verification
- Run YOLO training on the package segmentation dataset and review validation metrics.
- Verify gate false-negative rate on package images is acceptably low.
- Run Python benchmark on a mixed package/non-package test set.
- Compare:
  - current classifier-only latency
  - YOLO-only latency
  - gated total latency
  - classifier skip rate
- Install/run the Android app on a target smartphone and record on-device timings.
- Manually test images with packages, no packages, cluttered backgrounds, and multiple packages.

## Decisions
- Use the newest available Ultralytics segmentation model, not a fixed YOLO26 checkpoint if that is unavailable/outdated.
- Use confidence/detection output for the package/no-package gate; segmentation masks are not required in the smartphone pipeline.
