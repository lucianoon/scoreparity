# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `scoreparity compare` and `scoreparity init` commands; Python API `compare`, `compare_files`.
- Strict YAML configuration (unknown keys are errors, YAML 1.1 exponent strings such as `1e-5`
  are read as numbers) with presets `exact`, `float-noise` and `quantization`.
- Gates: `coverage`, `nonfinite`, `max_abs_diff`, `quantile_abs_diff`,
  `mean_diff_equivalence` (TOST, global and per segment), `decision_flips`, `top_k_overlap`,
  `auc_difference` (paired DeLong).
- Input alignment that refuses duplicated ids, missing columns, non-numeric scores and id type
  mismatches; labels and segments can come from a separate context table.
- JSON report with `schema_version` (strict JSON: NaN is written as null) and Markdown report;
  AUC is summarised whenever labels are given.
- Lossless CSV reading (`float_precision="round_trip"`).
- Project skeleton: packaging, typed package marker, CLI entry point with stable exit codes,
  CI on Linux, Windows and macOS for Python 3.10 to 3.13, minimum-dependency job and wheel
  smoke test.
