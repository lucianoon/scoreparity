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

> Status: early development (0.1.0.dev). The CLI exit codes and the JSON report schema are
> designed as stable contracts, but may still change before 0.1.0.

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
(missing columns, duplicated ids, unreadable files). The Markdown report goes to stdout; use
`--json` and `--markdown` to write them to files.

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
report.passed          # bool
report.failed_gates    # ["max_abs_diff", ...]
report.to_json()       # stable, versioned schema (schema_version = 1)
report.to_markdown()   # for a PR comment or CI job summary
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
