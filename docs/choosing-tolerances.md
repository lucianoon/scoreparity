# Choosing tolerances

An equivalence test is only as good as its margin. Too tight, and harmless reruns fail until
people stop trusting (or start disabling) the gate. Too loose, and real changes slip through.
There are two sound ways to choose a margin, and the best configurations use both.

## 1. Measure the noise floor (`scoreparity noise`)

Every scoring system has a noise floor: the amount its scores move when *nothing meaningful*
changed. Typical sources:

| Source | Typical size (float32 model, probability scores) |
|---|---|
| Different batch size or batch composition | 1e-9 to 1e-7 |
| Different number of CPU threads / BLAS build | 1e-9 to 1e-7 |
| CPU vs GPU, or GPU vs another GPU model | 1e-7 to 1e-5 |
| Non-deterministic GPU kernels (atomics, cuDNN autotuning) | 1e-7 to 1e-5 |
| float32 storage of float64 scores | about 3e-8 relative |
| bfloat16 / fp16 inference | 1e-3 to 1e-2 (this is quantization, not noise) |

A gate tighter than the noise floor fails on reruns; a gate far looser than it hides real
changes. So measure it: score the **reference** model again under conditions that should not
matter and let `scoreparity` turn the measurements into a configuration.

```bash
# the same model, three harmless reruns: batch size 1, 4096 and a different machine
scoreparity noise --reference ref.parquet \
  --replicate rerun_bs1.parquet --replicate rerun_bs4096.parquet --replicate rerun_gpu2.parquet \
  --id customer_id --label churned --segment plan --out scoreparity.yaml
```

How the suggestions are built:

- **Continuous gates** (`max_abs_diff`, `quantile_abs_diff`, `mean_diff_equivalence`,
  `auc_difference`): worst value over the replicates × `--safety` (default 3), rounded *up* to
  two significant digits. For the mean and the AUC the measured quantity is the far edge of the
  confidence interval, globally and in every gated segment, so small segments with wide
  intervals are covered.
- **Discrete gates** (`decision_flips`, `top_k_overlap`, and the AUC again): a few replicates
  that showed *zero* flips do not prove flips are impossible. The suggestion is therefore never
  stricter than what the `max_abs_diff` tolerance `M` already allows: only rows within `M` of a
  threshold can flip, only rows within `2M` of the top-k cut can swap, and only
  positive/negative pairs within `2M` can change the AUC. These counts are exact, computed from
  the reference scores.
- With **zero** measured noise (a deterministic pipeline), the suggestion is the `exact`
  behaviour: `max_abs_diff: 0` and a mean margin of 1e-12.

Use **three or more** replicates, and make them as different as the real-life variation you
want to tolerate (if production may run on another instance type, include one). The project's
test suite checks the calibration by simulation: with three replicates and the default safety
factor, 300 out of 300 fresh harmless reruns pass across noise levels from 1e-8 to 1e-5, while
a shift of ten times the worst noise always fails.

## 2. Ask what the business can absorb

The noise floor says what is *detectable*; the use of the score says what *matters*. Check that
the measured tolerances are also acceptable from the consumer's point of view:

| How the score is used | Gate that protects it | Questions to ask |
|---|---|---|
| A cut-off decides an action (approve, contact, block) | `decision_flips` at that threshold | How many changed decisions per 100k is acceptable? |
| The top k% get an action (retention campaign, review queue) | `top_k_overlap` at that k | How much churn in the target list is acceptable? |
| The probability itself is consumed (pricing, expected loss) | `max_abs_diff`, `quantile_abs_diff`, `mean_diff_equivalence` | What error in expected value is acceptable? |
| Model quality is what matters (offline evaluation) | `auc_difference` | What AUC loss would anyone notice? |
| A segment has its own owner or regulation | `segments` + `mean_diff_equivalence` | Can a segment move while the total stays flat? |

If the business tolerance is **looser** than the noise-based one, keep the noise-based one: the
gate then catches changes long before they matter. If it is **tighter**, the system cannot
guarantee it as built; that is a finding (make the pipeline deterministic, fix the hardware,
or accept the risk explicitly).

## Presets as starting points

`exact`, `float-noise` and `quantization` are reasonable defaults for probability-like scores
in [0, 1], useful before any measurement exists. Replace them with a measured configuration as
soon as you can, and keep that configuration in version control: a margin chosen after seeing
the result is not a test.

## Statistical fine print

- `mean_diff_equivalence` and `auc_difference` are equivalence tests (TOST): the (1 − 2α)
  confidence interval must lie inside ±margin. "No significant difference" is never used as
  evidence of equivalence.
- Requiring every segment to pass is an intersection-union test: the overall false-equivalence
  rate stays at or below α without multiplicity corrections. Segments smaller than
  `min_segment_size` are reported but not gated.
- With very large data the confidence intervals become tiny, so the margin, not the sample
  size, decides the outcome. That is intended: the margin is a statement about what matters.
