# Add trustworthy benchmark results

Keep hardware/workload claims separate from accuracy and safety claims. The
checked-in Mac and Tesla T4 results are real measurements. Accuracy and vehicle
safety remain unvalidated. Do not fill gaps with expected numbers.

## Current evidence

| Experiment | Hardware/workload | Evidence | Status |
|---|---|---|---|
| CPU profiles | Apple M1, 2 Torch threads, repeated 1280×720 sample | [Raw traces/report/source archive](../benchmarks/mac_m1_cpu/README.md) | Measured, 60 frames/profile |
| GPU profiles | Colab Tesla T4, 2 host threads, repeated 1280×720 sample | [Raw traces/report/session logs](../benchmarks/colab_t4_sample_20260930T202529Z/README.md) | Measured, 300 frames/profile |
| Indian-road accuracy | Held-out annotated driving footage | Dataset/training recipes only | Not yet established |
| Vehicle-level safety | Closed-loop scenarios and deployment hardware | Experimental rule/simulator interfaces | Not established |

The CPU reports were collected before CUDA instrumentation was added. Their
`git_dirty: true` records are supplemented by the exact measured source archive;
do not rewrite historical source hashes to match a newer checkout.

The T4 ran clean source commit `aa018dc54f538d314ff365b4c2a9943e71689822`.
Its sample bytes and checkpoint hashes/revisions match the CPU experiment, but
sample counts, packages and synchronization differ. No paired hardware speedup
or held-out accuracy improvement is claimed.

## Check the exported run

Before publishing, inspect all three JSON files and their corresponding CSVs:

1. `protocol.device` must identify the requested device and `environment.gpu`
   must identify the actual GPU for CUDA runs. Confirm source SHA and dimensions.
2. Frame count must equal frames-per-repeat × repeats. All host-thread values
   must match the requested budget; all measured latencies must be finite.
3. Full profiles must show successful OWLv2 completions during every measured
   repeat, with no recorded inference error. The runner checks this automatically.
4. Source/config/code hashes and checkpoint revisions must remain intact. Record
   the exported commit, full dependency list and GPU driver information as well.
5. Confirm rendering, warmup, frame count, repetitions, resolution and source are
   identical across profiles before making a profile comparison. Background OWL
   is deliberately disabled in this experiment; changing it changes freshness.
6. Review previews, empty-mask counts and scenario coverage. Runtime success is
   not prediction correctness. An unknown signal state does not establish that
   no light was present; annotations are required for that claim.

The summarizer checks matching protocols and input hashes. Source paths in
exports are local paths and may need redaction for public sharing; never include
tokens, private footage, model caches or personal notebook outputs.

## Add a new artifact directory

Extract the exported ZIP locally. Move its report/JSON/CSV/PNG/session metadata
into a new directory such as `benchmarks/colab_t4_sample_2026-10-01/` only if the
actual session used a T4 on that date. Choose the name from observed hardware,
workload and run date; do not reuse that example blindly.

Do not overwrite `benchmarks/mac_m1_cpu/`. The two experiments use different
devices and currently different sample counts. Commit all three profile reports,
their raw traces, chart, previews and session metadata together. Add a link in
the README alongside the CPU report. Retain `accuracy` fields as null unless an
actual annotated evaluation produced those values.

Record these fields in a GPU results table:

| Field | Unit/meaning | Source |
|---|---|---|
| Profile + GPU | Actual configuration and assigned hardware | `profile`, `environment.gpu.name` |
| FPS | Aggregate completed frames/elapsed wall time | `throughput_fps` |
| p50/p95/p99 | End-to-end frame milliseconds | `latency_ms.wall_ms` |
| Peak host RSS | Whole-process MiB | `process_peak_rss_mb` |
| Peak allocated/reserved VRAM | Post-warmup PyTorch allocator MiB | `gpu_peak_allocated_mib`, `gpu_peak_reserved_mib` |
| Sample count / warmup | Measurement protocol | `protocol` |
| Source / rendering / device | Workload definition | `source`, `protocol` |

Headline FPS must link to the supporting artifact directory. Do not compare
`full512` to `core256` as an equivalent-output optimization: the latter disables
OWL and changes segmentation resolution. Keep CPU and GPU rows explicitly
labelled, even when one is much faster.

## Make fair comparisons

For a hardware comparison, rerun identical source bytes, model weights, profile,
render setting, resolution, repeats and frame counts on both devices. Preserve
package versions and timing instrumentation notes. For an optimization study,
capture the baseline before changing code and evaluate both prediction quality
and speed on the same held-out inputs. The present table compares operating
profiles, not a proven before/after code-only speedup.

Use longer videos and enough repetitions to expose variability. Report per-repeat
FPS and all percentiles, not only the best run. Watch thermal/load/session effects.
CUDA stage synchronization and CPU device execution differ by design; no overlap
or batch speedup is being measured. A small-sample p99 is descriptive, not an SLA.

## What would justify a stronger claim?

| Claim | Evidence needed |
|---|---|
| Real-time on target hardware | Defined frame-rate/deadline, p95/p99 under decode/queue/render load, dropped-frame and stale-result behavior |
| Accurate for Indian roads | Annotated held-out split, class-wise precision/recall/mAP, road IoU, domain/scenario slices and failure examples |
| Robust multi-object tracking | Moving footage with IDs, ID switches and tracking metrics across occlusion/mixed traffic |
| Better optimization | Paired baseline/candidate runs, identical workload plus accuracy/freshness tradeoffs |
| Deployment-ready perception | Target hardware profiling, resource bounds, monitoring, error recovery and reproducible packaging |
| Vehicle-level ADAS safety | A separate safety process, physical calibration, closed-loop validation and applicable requirements—not a notebook FPS result |

For a strong personal project, the compelling story is evidence-backed engineering:
correctness fixes, documented tradeoffs, raw traces, reproducible tests and candid
failure analysis. Keep measured findings linked to their evidence and to the next
validation task; this does not turn prototype capabilities into industry certification.
