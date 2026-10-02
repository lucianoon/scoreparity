# scoreparity

**Prove that a change to your ML system did not change what your model outputs.**

Refactors, library upgrades (PyTorch, CUDA, scikit-learn), moving from CPU to GPU, quantization,
rewriting preprocessing, migrating from a notebook to a pipeline or between platforms: all of
these are supposed to *preserve* model behaviour. In practice they are usually checked by eye,
often by comparing an aggregate metric such as AUC. That is not enough:

- AUC can stay identical while individual scores move enough to flip decisions.
- A "no significant difference" t-test passes precisely when the evidence is weakest.
- A global mean difference of zero can hide segments that moved in opposite directions.

`scoreparity` compares the scores of a **reference** and a **candidate** version on the same
records and answers one question with sound statistics: *are they equivalent within the
tolerance you declared up front?* It is model-agnostic: it only needs two tables of scores.

> Status: 0.1 (alpha). The CLI exit codes and the JSON report schema (version 1) are stable
> contracts; other APIs may change in minor releases until 1.0.

## Quick start

```bash
pip install "scoreparity[parquet]"   # not on PyPI yet: pip install git+https://github.com/lucianoon/scoreparity

scoreparity compare \
  --reference scores_v1.parquet \
  --candidate scores_v2.parquet \
  --id customer_id --label churned --segment plan --segment region \
  --preset float-noise
```

Exit codes: **0** equivalent, **1** not equivalent, **2** the comparison could not be made
(missing columns, duplicated ids, unreadable files). The Markdown report goes to stdout.

| Output | Flag | For |
|---|---|---|
| JSON | `--json r.json` | machines; versioned schema, re-renderable with `scoreparity render` |
| Markdown | `--markdown r.md` | pull-request comments and CI job summaries |
| HTML | `--html r.html` | humans: a self-contained page (no external resources) with charts of how large the differences are and which segments moved, in light and dark mode |
| JUnit XML | `--junit junit.xml` | any CI test UI (GitHub, GitLab, Jenkins, Azure DevOps): one test per gate |

A JSON report produced elsewhere (a pipeline job, a notebook) can be rendered later:
`scoreparity render r.json --html r.html --junit junit.xml`.

## GitHub Action

```yaml
permissions:
  contents: read
  pull-requests: write   # for the PR comment; optional

steps:
  - uses: actions/checkout@v7
  - run: python score.py --out candidate.parquet       # however you produce scores
  - uses: lucianoon/scoreparity@v0.1.0                 # or pin the commit SHA
    with:
      reference: baseline/scores.parquet
      candidate: candidate.parquet
      config: scoreparity.yaml
```

The action writes the report to the job summary, keeps one up-to-date comment on the pull
request, uploads JSON/Markdown/HTML/JUnit as an artifact and fails the job when the candidate is
not equivalent (`fail-on-mismatch: "false"` turns that into a warning; input errors always fail).
Outputs: `verdict` (PASS/FAIL/ERROR), `exit-code`, `report-dir`. To try it locally, generate
example data with `python examples/make_example_data.py`.

Real output, comparing a notebook model with its pipeline migration (left) and the same
model run with bfloat16 autocast (right):

| Gate | Migration | bfloat16 |
|---|---|---|
| `max_abs_diff` | pass (1.8e-7) | **FAIL** (8.7e-3) |
| `quantile_abs_diff` (q0.99) | pass (1.2e-7) | **FAIL** (4.2e-3) |
| `mean_diff_equivalence` (TOST, per segment) | pass | **FAIL** in every segment |
| `decision_flips` at 0.5 | pass (0) | **FAIL** (1.7e-4) |
| `top_k_overlap` (top 10%) | pass (1.0) | **FAIL** (0.998) |
| AUC difference | 0 | -8.5e-6, 90% CI contains 0 |
| Paired t-test on scores (not a gate) | — | p = 0.80, i.e. "no significant difference" |

## Python API

```python
import scoreparity as sp

cfg = sp.from_dict({"preset": "float-noise", "columns": {"id": "customer_id"}})
report = sp.compare(reference_df, candidate_df, cfg)
report.passed  # bool
report.failed_gates  # ["max_abs_diff", ...]
report.to_json()  # stable, versioned schema (schema_version = 1)
report.to_markdown()  # for a PR comment or CI job summary
```

## Configuration

`scoreparity init` writes a commented `scoreparity.yaml`. Keep it in version control next to
the code it protects: the tolerance is decided *before* the comparison, not after.

```yaml
version: 1
preset: float-noise          # exact | float-noise | quantization
columns: {id: customer_id, score: score, label: churned}
segments: [plan, region]
gates:
  max_abs_diff: {max: 1.0e-5}
  top_k_overlap: {k_pct: [1, 10], min_overlap: 0.99}
  auc_difference: {margin: 0.001}
  decision_flips: null        # null disables a gate from the preset
```

Unknown keys are errors: a misspelled gate silently disabled would be worse than no gate.

| Gate | Passes when | Use it for |
|---|---|---|
| `coverage` (default on) | candidate scored at least `min` of the reference ids | dropped rows |
| `nonfinite` (default on) | NaN/inf appear on both sides or neither | numerical blow-ups |
| `max_abs_diff` | worst \|candidate − reference\| ≤ `max` | strict migrations |
| `quantile_abs_diff` | the `q` quantile of \|diff\| ≤ `max` | tolerating rare outliers |
| `mean_diff_equivalence` | TOST: the (1−2α) CI of the mean diff lies inside ±`margin`, globally and in every segment with ≥ `min_segment_size` rows | systematic shifts |
| `decision_flips` | share of rows changing side of each threshold ≤ `max_rate` | classifiers with a cut-off |
| `top_k_overlap` | overlap of the top-k% sets ≥ `min_overlap` | ranking, targeting |
| `auc_difference` | paired DeLong (1−2α) CI of ΔAUC inside ±`margin` | model quality |

## Class labels and probability vectors

Classifiers (including LLM-based ones) often output a **class** or a **vector of class
probabilities** rather than one score. Set `output.type`:

```yaml
version: 1
output: {type: label}              # or: {type: probabilities, prob_prefix: "p_"}
columns: {id: message_id, score: intent, truth: human_intent}   # `score` = the label column
segments: [channel]
gates:
  label_agreement: {min: 0.99}     # lower bound of the agreement, globally and per segment
  transitions: {max_rate: 0.005}   # no class may lose more than 0.5% of its rows to another
  class_prevalence: {margin: 0.005}
  kappa: {min: 0.97}               # agreement beyond chance (matters with a dominant class)
  quality_difference: {metric: macro_f1, margin: 0.01}   # needs columns.truth
```

| Gate | Passes when | Method |
|---|---|---|
| `label_agreement` | the one-sided lower bound of the share of rows with the same class ≥ `min` | Clopper-Pearson (exact) |
| `transitions` | for each class with ≥ `min_class_size` rows, the upper bound of the share of its rows that moved to each other class ≤ `max_rate` | Clopper-Pearson, intersection-union |
| `class_prevalence` | the (1−2α) CI of each class's paired share difference lies inside ±`margin` | Tango score interval |
| `kappa` | the lower bound of Cohen's kappa between the versions ≥ `min` | Fleiss–Cohen–Everitt variance |
| `quality_difference` | the CI of accuracy or macro-F1 difference against `truth` lies inside ±`margin` | Tango (accuracy), bootstrap (macro-F1) |
| `tv_distance` (probabilities) | a quantile of the per-row total variation distance ≤ `max` | descriptive |

For `probabilities`, `max_abs_diff`, `quantile_abs_diff` and `mean_diff_equivalence` apply to
every class column (all must pass), class gates use each row's most probable class, and the
`exact`/`float-noise`/`quantization` presets keep only the gates that apply.

Class gates make a statement about the **population** the rows were sampled from, so they need
enough rows: proving "at most 1% of a class moves" with zero observed moves needs 299 rows of
that class. When there are not enough, the gate fails and the report says how many rows are
needed (see [docs/choosing-tolerances.md](docs/choosing-tolerances.md#how-many-rows)).

## Calibrated tolerances: `scoreparity noise`

Guessing a margin is the weak spot of every equivalence test. Instead, score the **reference**
model again under conditions that should not matter (another batch size, machine or thread
count) and let `scoreparity` measure the noise floor and write the configuration:

```bash
scoreparity noise --reference ref.parquet \
  --replicate rerun1.parquet --replicate rerun2.parquet --replicate rerun3.parquet \
  --id customer_id --label churned --segment plan --out scoreparity.yaml
```

Tolerances are the worst measured noise × a safety factor (default 3), and the discrete gates
(flips, top-k, AUC ordering) are never set stricter than what the `max_abs_diff` tolerance
already allows. In the test suite's simulations, 300 of 300 fresh harmless reruns pass and a
change of ten times the noise always fails. See
[docs/choosing-tolerances.md](docs/choosing-tolerances.md) for the method and for how to
combine it with what the business can absorb.

## More documentation

- [Choosing tolerances](docs/choosing-tolerances.md)
- [Integrations](docs/integrations.md): GitHub Actions, GitLab CI / Jenkins / Azure DevOps
  (JUnit), Amazon SageMaker Pipelines, generic orchestrators
- [Report JSON schema](schema/report-v1.schema.json): every report is validated against it in
  the test suite

## Statistical notes

- **Equivalence, not difference.** TOST (Schuirmann, 1987) rejects "the difference is at least
  `margin`". Its false-equivalence rate is at most α; the test suite checks this by simulation.
- **Segments** are combined as an intersection-union test (Berger, 1982): every segment must
  pass at level α, which keeps the overall error at or below α without a multiplicity correction.
- **AUC** uses DeLong et al. (1988) with the O(n log n) algorithm of Sun & Xu (2014); the
  standard error is validated against a paired bootstrap and the interval coverage by simulation.
- **Reading files never invents differences**: CSV floats are parsed with pandas'
  `round_trip` parser (the default parser returns about a quarter of float64 values 1 ulp off).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). CI runs on Linux, Windows and macOS for Python 3.10 to
3.13, plus a job with the lowest declared dependency versions.

## License

Apache-2.0. See [LICENSE](LICENSE).
