# RoadSense India — CPU benchmark

Real YOLOv8n, SegFormer-B0 and (for full profiles) OWLv2 inference. No mock model timings.

Hardware: Apple M1; device: CPU; PyTorch threads: 2; input: 1280×720.

Workload: Repeated letterboxed image: bus.jpg. Each profile uses a separate process, 2 repeats of 30 measured frames, and 10 warmup frames per repeat. Frames include input copying, perception, tracking, segmentation, signal classification, anomalies, decisions and rendering. Image loading/letterboxing, preview export and model initialization are excluded from warm throughput and reported separately where applicable.

| Profile | OWLv2 | SegFormer input | FPS | p50 frame ms | p95 frame ms | p99 frame ms | Peak process RSS MiB |
|---|---|---|---:|---:|---:|---:|---:|
| full512 | every 10 frames | 512×512 | 0.85 | 428.4 | 7804.2 | 7954.5 | 2226 |
| full256 | every 10 frames | 256×144 | 1.07 | 176.6 | 7552.7 | 7985.9 | 2229 |
| core256 | disabled | 256×144 | 5.13 | 180.9 | 289.2 | 328.8 | 440 |

![Measured throughput and tail latency](comparison.png)

## Where time goes

| Mean stage latency, ms | full512 | full256 | core256 |
|---|---:|---:|---:|
| detection_ms | 59.92 | 60.98 | 66.94 |
| zero_shot_ms | 744.04 | 750.17 | 0.00 |
| segmentation_ms | 368.00 | 113.58 | 119.07 |
| tracking_ms | 0.37 | 0.36 | 0.40 |
| signal_ms | 0.00 | 0.00 | 0.00 |
| anomaly_ms | 0.68 | 4.00 | 4.12 |
| decision_ms | 0.03 | 0.07 | 0.08 |
| visualization_ms | 1.41 | 3.59 | 3.80 |

## Interpretation

Synchronous OWLv2 dominates mean and tail latency. Its mean cost is amortized over skipped frames; p95 captures the periodic stalls. Reducing segmentation resolution improves network latency, but does not remove zero-shot stalls. core256 is a different operating profile, not a speedup with identical outputs.

The full512 workload had 60/60 frames with an empty road mask; compact profiles can produce different masks and fitted boundaries. This is evidence of sensitivity to preprocessing/resolution, not evidence that either profile is accurate. Signal classification has no detected traffic lights in this sample, and road-anomaly analysis takes its empty-mask path in full512. Scenario tests cover rules separately, but moving-road footage with ground truth is needed for mAP, road IoU and MOTA. Those accuracy metrics are explicitly null in the JSON reports.

Process RSS includes Python, libraries and models. It is a process peak, not GPU memory or steady-state resident memory. Host load is recorded in each JSON and can change timings; p99 from 60 frames is descriptive, not an SLA.

## Reproduce

```bash
pip install -r requirements.txt -c constraints-tested.txt
python scripts/benchmark_pipeline.py --profile all --source sample --frames 30 --warmup 10 --repeats 2 --threads 2 --output runs/benchmark/result.json
python scripts/summarize_benchmark.py --directory runs/benchmark
```

Models download on the first run. After caching them, set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for an offline repeat. For headless cost use --no-render; for real footage use --source /path/to/video.mp4 with enough frames for warmup and measurement.

## Evidence

The JSON hashes describe the exact, uncommitted measurement source, not necessarily
the latest repository code. [Measured source archive](measured_source.tar.gz)
preserves `src/` and the original benchmark runner. Extract it into a separate
scratch directory, not over your working checkout. CUDA instrumentation was added
after these CPU runs; the original records and measured values are unchanged.
Install `requirements-dev.txt` for the chart generator (Matplotlib is a reporting
dependency). `constraints-tested.txt` captures direct Mac versions, not a CUDA lock.

[full512 JSON](full512.json) · [full256 JSON](full256.json) · [core256 JSON](core256.json)

[full512 raw CSV](full512.csv) · [full256 raw CSV](full256.csv) · [core256 raw CSV](core256.csv)

Each JSON records source/config/code hashes, checkpoint revisions, versions, cold initialization, warmup, per-repeat FPS, OWL completion/error status and all measured frame samples.
