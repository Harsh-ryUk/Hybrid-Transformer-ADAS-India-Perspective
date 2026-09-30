# RoadSense India — system/model card

## Intended use

Road-scene research, reproducible latency experiments, annotated video demos and
simulation-oriented rules. This repository does not establish vehicle-level
safety or Level 4 autonomy. Legacy names remain for source/API compatibility.

## Components and evidence

| Component | Implementation | Validation here | Limitation |
|---|---|---|---|
| Detection | COCO YOLOv8n; actual labels; category thresholds | Real runtime; custom-taxonomy regression | India-specific fine-tuned weights not measured |
| Segmentation | ADE20K SegFormer-B0; road label 6; RGB input | Processor equivalence; real runtime; mask tests | General scene model; not lane-marking segmentation |
| Zero-shot | OWLv2 road-object text prompts | Real full benchmarks; scheduling/filter tests | Supplemental display; periodic slow inference |
| Tracking | Kalman + IoU + Hungarian | Lifecycle/association tests | No learned appearance; not full DeepSORT |
| Signals | HSV crop heuristic | Tests/orchestration; no lights in sample | No signal relevance reasoning |
| Anomalies | Image velocity/proximity/contrast rules | Synthetic scene tests | No metric depth or calibrated TTC |
| Decisions | Priority rules; simulation controls | Expected-action tests | Pixels are not physical stopping distances |

RGB preprocessing follows the official
[SegFormer input convention](https://huggingface.co/docs/transformers/model_doc/segformer).
Road-mask coverage is a geometric fraction, not calibrated confidence.
Capture input is first reduced to 256×144 even for the 512×512 network profile.

## Data and measurement

[CPU evidence](../benchmarks/mac_m1_cpu/README.md) and
[Tesla T4 evidence](../benchmarks/colab_t4_sample_20260930T202529Z/README.md) use one letterboxed, repeated
Ultralytics sample image at 1280×720. It exercises real inference but lacks motion,
diverse road scenes and annotations. Throughput includes per-frame input copying
and rendered processing; model construction and warmup are separate. Memory is
peak process RSS; GPU records additionally report post-warmup PyTorch allocated/
reserved memory. The T4 measured full512/full256/core256 at 8.05/8.06/19.65 FPS,
with p95 wall latency of 777.0/785.6/71.7 ms over 300 frames/profile. These are
synchronized single-stream sample measurements, not target-device deadlines.

The 512×512 profile produced no road mask on this sample. Smaller inputs changed
masks/boundaries. This is an observation, not an accuracy comparison. The sample
does not exercise every anomaly branch or traffic-light state.

## Failure modes

- COCO pretraining does not establish recognition of auto-rickshaws/overloaded vehicles.
- Road masks and fitted edges can fail under padding, lighting, occlusion or domain shift.
- IoU-only tracking can switch identities across occlusion/overlap.
- Cached OWLv2 boxes are older observations with exposed source age, without motion compensation.
- Pixel distance/velocity depends on camera geometry and ego motion.
- Missing masks reduce road-anomaly information; they do not establish clear road.
- Experimental rule commands have no complete vehicle safety envelope.

## Next validation

Use an annotated held-out Indian-road set to measure class-wise precision/recall,
road IoU, tracking ID switches/MOTA, event precision/recall and decision errors.
Compare resolution/scheduling profiles on identical examples. Then measure
deployment hardware with decode, queueing, cold start, peak/steady memory and
stale-observation rates. CARLA/ROS closed-loop behavior was not tested here.
