# Engineering notes

## Measurement protocol

Periodic slow models create spikes that mean FPS hides. The benchmark reports
p50/p95/p99 and aggregate frames/elapsed time, and verifies that enabled OWLv2
completed inference. Full profiles retain the same modules; `core256` deliberately
removes supplemental zero-shot inference. Their output/memory budgets differ.

Inspect every tenth-frame stall in raw traces. CPU measurements used two 30-frame
repeats; the T4 used three 100-frame repeats. Neither establishes a reliability/
tail-latency guarantee. Image loading
and letterboxing precede measurement; per-frame copy/decode and processing are
included. Cold construction and per-repeat warmup are separately recorded.

## Operational contracts

`ADASPipelineL4.process_frame(frame, render=False)` accepts nonempty H×W×3
uint8 BGR input and returns the input plus metrics without rendering. Rendering
returns an annotated copy. Missing/invalid configs raise instead of switching
profiles silently. CPU/CUDA device resolution is shared across modules; unavailable
CUDA falls back to CPU with a log. MPS is not currently a supported pipeline device.

Use the pipeline as a context manager or call `close()` to drain worker inference.
OWL status exposes completions, model availability, worker activity, errors and
observation age. Cached loading uses a model/device key; detectors retain their
own model references. Timing uses a monotonic clock; the HUD displays rolling FPS
from completed frames rather than reading an unfinished latency record.

## Tests and reproducibility

Offline tests check algorithms, expected scene rules, stage accounting, RGB input,
custom label preservation, invalid inputs, bounded metadata and one-job scheduling.
Legacy v2 tests stub model outputs; the benchmark separately runs real models.
Hosted CI passed on Linux/Python 3.10 and 3.11. Local checks used
macOS/Python 3.9.6. Tested constraints pin direct runtime versions, not a complete
portable dependency lock. The CUDA runner now synchronizes stage boundaries,
rejects unavailable devices and reports PyTorch allocated/reserved memory.
These contracts have mock-based regression tests plus a completed physical
Tesla T4 experiment (Python 3.13.15, Torch 2.11.0+cu128); all 70 project tests
passed in that session before real-model benchmarking. Closed-loop vehicle
behavior remains unvalidated.

Use [the Colab notebook](../notebooks/colab_benchmark.ipynb) and
[GPU guide](COLAB.md) to reproduce the [published T4 experiment](../benchmarks/colab_t4_sample_20260930T202529Z/README.md).
The full-profile OWLv2 stalls remain the main latency bottleneck. A smaller
SegFormer input alone did not materially improve aggregate full-profile FPS.
Keep the existing CPU raw records:
their exact source is archived alongside them. The [results protocol](RESULTS.md)
explains separate artifact directories and the evidence required for comparisons.

## Portfolio talking points

- Profiled a multi-model vision pipeline end to end, including periodic tail-latency spikes.
- Compared resolution/module-budget profiles using checkpoint/hash provenance and raw traces.
- Corrected RGB, taxonomy and road-mask bugs before trusting optimization numbers.
- Added offline regression tests, CI, lifecycle management and input contracts.
- Distinguished measured runtime from unmeasured driving/deployment capability.

Avoid presenting pretrained weights as IDD-fine-tuned, IoU association as
appearance-based DeepSORT, or simulation commands as validated autonomous driving.
