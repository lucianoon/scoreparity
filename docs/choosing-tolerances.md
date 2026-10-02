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

### Class labels (LLM classifiers)

`scoreparity noise --output-type label` does the same for class outputs: rerun the reference
classifier on the same rows (three or more times) and it suggests `label_agreement`,
`transitions`, `class_prevalence`, `kappa` and, with `output.normalize.allowed`, `invalid_rate`.

- The measured quantity is the unfavourable **confidence bound** each gate decides on, not the
  point estimate, globally and in every gated segment or class. Because those bounds already
  include the sampling uncertainty of a run of that size, the default safety factor is 1.75
  rather than 3.
- `transitions` gets **one limit per source class** (`per_class`). In a typical classifier a
  small class receives stray answers from the big ones, so its rows move several times more
  often than those of a big class; a single limit would be set by that class and hide real
  changes elsewhere.
- Calibration was chosen by simulation (five classes, 2,000 to 10,000 rows, 0.5% to 2% of the
  answers resampled per run, three replicates): at least 97% of fresh reruns pass, and a
  candidate that moves 5% of one class to another fails every time. Three times the noise is
  caught about half of the time; `--safety 1.5` catches it more often, at the price of more
  false alarms on reruns.

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

## How many rows?

The score gates describe the rows you give them. The **class gates** (`label_agreement`,
`transitions`, `class_prevalence`, `kappa`, `quality_difference`) make a statement about the
population the rows were sampled from, which is what matters when the rows are a sample (for
example, 2,000 messages re-classified by a new LLM version). A sample can only prove a tight
tolerance if it is large enough, even when it shows no difference at all.

Rows needed when **zero** disagreements are observed (one-sided 95%, `alpha: 0.05`):

| Claim | Rows needed |
|---|---|
| agreement ≥ 95% (or a class loses ≤ 5% of its rows) | 59 |
| ≥ 99% / ≤ 1% | 299 |
| ≥ 99.5% / ≤ 0.5% | 598 |
| ≥ 99.9% / ≤ 0.1% | 2,995 |
| ≥ 99.95% / ≤ 0.05% | 5,990 |

For `transitions` the count is **per class**: a class with fewer rows cannot be verified at
that tolerance. The report names those classes and the number of rows they need. You can then
collect more rows, loosen the tolerance, or deliberately exclude small classes by raising
`min_class_size` (they are still visible in the transition matrix). Observed disagreements
raise the numbers above, so plan with headroom.

`scoreparity plan-sample` does that planning exactly. Give it the tolerance and the rate you
expect (from a noise run, or from experience) and it returns the smallest sample from which the
gate passes with the requested power (default 80%), plus the cost if you give a price per row:

```bash
# transitions at 1% when about 0.2% of a class moves, smallest class = 5% of the rows
scoreparity plan-sample --max-rate 0.01 --expected-rate 0.002 --class-share 0.05
#   rows needed: 773 per class ... 15,460 rows in total
```

The power of a binomial test is a sawtooth in n: one more row can lower it slightly. The plan
reports the size from which every larger sample keeps the power, and also the first size that
reaches it, which may be smaller.

## Statistical fine print

- `mean_diff_equivalence` and `auc_difference` are equivalence tests (TOST): the (1 − 2α)
  confidence interval must lie inside ±margin. "No significant difference" is never used as
  evidence of equivalence.
- Requiring every segment to pass is an intersection-union test: the overall false-equivalence
  rate stays at or below α without multiplicity corrections. Segments smaller than
  `min_segment_size` are reported but not gated.
- With very large data the confidence intervals become tiny, so the margin, not the sample
  size, decides the outcome. That is intended: the margin is a statement about what matters.
