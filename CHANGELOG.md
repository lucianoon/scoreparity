# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-02

### Security

- Third-party actions pinned to commit SHAs in the workflows and in the composite action;
  least-privilege `permissions` per job; checkout credentials not persisted.
- CI audits locked dependencies for known vulnerabilities (`pip-audit`, also weekly), lints
  workflows with `zizmor`, and runs CodeQL (`security-extended`) on Python and Actions code.
- `SECURITY.md` with private vulnerability reporting.

### Added

- `scoreparity noise`: measures the noise floor from replicate runs of the reference model and
  writes a configuration whose tolerances cover it (worst noise × safety factor; discrete gates
  never stricter than what the `max_abs_diff` tolerance allows). Generated YAML escapes user
  data (file and column names).
- JSON Schema for the report (`schema/report-v1.schema.json`), enforced by the test suite.
- Documentation: `docs/choosing-tolerances.md`, `docs/integrations.md`.
- Self-contained HTML report (`--html`): order-of-magnitude histogram of the differences and a
  per-segment confidence-interval chart, light and dark mode, all user data escaped.
- JUnit XML report (`--junit`): one test case per gate.
- `scoreparity render`: re-render a saved JSON report; exit code mirrors its verdict.
- GitHub Action (`action.yml`): job summary, sticky pull-request comment, report artifact,
  `verdict`/`exit-code`/`report-dir` outputs; inputs never interpolated into shell scripts.
- `examples/make_example_data.py`: deterministic example data for demos and the action self-test.
- `summary.abs_diff_histogram` in the JSON report.
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

[Unreleased]: https://github.com/lucianoon/scoreparity/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lucianoon/scoreparity/releases/tag/v0.1.0
