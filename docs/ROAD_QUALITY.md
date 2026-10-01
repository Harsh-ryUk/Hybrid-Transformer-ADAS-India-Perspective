# Inspect road regions and possible surface defects

This remains a research demo. SegFormer predicts semantic road regions, not
painted lanes. Dark patches are not confirmed potholes and rule outputs are not
vehicle controls validated for real use.

## What changed

- Removed the forced 256×144 capture resize. Network input now comes directly
  from the selected source crop; the network resize still uses its configured
  width/height, so choose a suitable profile rather than assuming aspect invariance.
- Interpolate logits before argmax, close small holes and keep the largest
  lower-image road component. No broad dilation or fabricated trapezoid fallback.
- Reject tiny road coverage, expose `road_status: unknown` and prevent default
  cruise when the pipeline explicitly reports missing road perception.
- Remove fitted road-edge polynomials from lane steering: no trained lane-marking
  detector is included. The HUD distinguishes road observation from lane validation.
- Use local dark contrast rather than global mean brightness for surface candidates.
  Erode mask borders, exclude tracked-object boxes, reject large/elongated patches
  and require consecutive-frame overlap before reporting a possible defect/shadow.
- Check animal ground contact at the bottom of its box. Disable wrong-side inference
  by default in the L4 pipeline because camera motion is not compensated.

## Configure the camera region

```yaml
segmentation:
  input_size: [512, 512]
  road_class_ids: [6]
  frame_roi: [0, 0, 1, 1]
  roi_top_fraction: 0.45
  min_road_coverage: 0.02
anomaly:
  wrong_side_enabled: false
  surface_min_frames: 3
```

`frame_roi` is normalized `[left, top, right, bottom]`. Masks map back to original
image coordinates; pixels outside the crop remain zero. The supplied
`configs/dashcam_quality.yaml` is specifically cropped for the vertical Short
`0ziIYyosuuE`; do not reuse that camera crop blindly. Inspect mask overlays after
changing camera framing, resolution or labels. ADE20K road is class 6; a separately
fine-tuned binary model needs its own label IDs.

```bash
python -m src.adas_pipeline_l4 --config configs/dashcam_quality.yaml --source video.mp4 --device cpu --headless --output result.mp4
python -m pytest -q tests/test_road_quality.py
```

## Evidence, finding and next validation

The supplied clip exposed empty masks after severe portrait-to-16:9 preprocessing.
Real-model spot checks showed that native-detail/cropped input can restore road
regions on sampled frames. This motivated a clip-specific quality profile and
regression tests for crop mapping, unknown-road behaviour, temporal confirmation,
tracked-object exclusion and mask-border rejection.

More nonempty masks are not evidence of better IoU. Largest-component filtering
can remove valid separated road regions; generic ADE20K weights can still confuse
road, runway, land and the bonnet. The next task is annotated Indian-road evaluation
and appropriate fine-tuning, plus a dedicated lane-marking model if lanes are needed.
Surface candidates need labelled defects and negative examples of shadows, lane
paint, wipers and reflections; no defect accuracy is claimed here. Consecutive-frame
IoU is not motion-compensated tracking and can miss moving candidates.

Historical M1/T4 sample traces and checkpoint/source hashes remain unchanged.
The current profile has a separate [paired T4 moving-video benchmark](../benchmarks/colab_t4_video_20261001/README.md):
6.95 → 11.05 FPS and 207 → 132 ms p95 wall latency. The surface filter uses the
supported-road bounding region with a 60-pixel halo instead of the full frame.
Kernel and thresholds are unchanged; fixture tests compare supported pixels
exactly. Recorded per-frame counts, coverage and actions match on 1782 paired
frames. This is latency/regression evidence, not better segmentation or defect
accuracy. A sampled preview is not a controlled speed study.

## Supplied clip check

On 2026-10-01, the Mac CPU decoded all 614 source frames and processed every fourth
frame (154 total), exporting a silent preview at approximately 15 FPS playback.
This sampling rate is not inference throughput. The quality profile reported
148 observed road regions and six unknown; all six unknown-road frames selected
simulation slow-down, never default cruise. The restored historical compact path
returned entirely empty masks on 37 of the same sampled frames. Crop, resolution,
postprocessing and validity thresholds differ; these counts do not measure IoU.

There were 12 frame-level surface-candidate alerts, without defect ground truth.
Some candidates lie near road boundaries/bonnet and may be false positives.
No pothole precision/recall or anomaly accuracy improvement is claimed. The
comparison JSON and silent videos remain in the local ignored run directory;
the third-party footage has not been published in this repository.

The initial quality change passed 87 offline regression tests. Subsequent
evaluation, optimization and evidence tests expand that suite; 25 focused tests
also passed on Colab. Fatal-error lint checks remain clean.
