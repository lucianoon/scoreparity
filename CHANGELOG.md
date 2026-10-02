# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-10-02

### Added

- Label normalisation for LLM outputs (`output.normalize`): stripping, lowercasing, synonym
  `map` and `allowed` classes; anything else (refusals, empty or malformed answers) becomes the
  `invalid` class and takes part in every class gate. Ground truth is normalised the same way.
- `invalid_rate` gate: one-sided Clopper-Pearson upper bound of the candidate's share of
  invalid answers.
- `scoreparity plan-sample`: exact binomial sample-size planning for the rate gates
  (`label_agreement`, `transitions`, `invalid_rate`), with per-class totals and cost estimates.
  The returned size keeps the requested power for every larger sample (binomial sawtooth).
- `examples: N` / `--examples N`: rows listed by id in the JSON, Markdown and HTML reports
  (class changes round-robin over the kinds of change; largest differences for scores and
  probabilities). Off by default.
- `scoreparity noise --output-type label`: noise floor of a classifier from reruns of the
  reference model; suggests the class gates from the worst confidence bound × `--safety`
  (default 1.75 for labels), with one transition limit per class. Calibrated by simulation.
- `gates.transitions.per_class`: a limit per source class.
- Replicas (`columns.replica`, `--replica`): several answers per id compared by majority class;
  the report shows unstable ids per version and how many changes fall on them. New `stability`
  gate (Tango upper bound of the increase in unstable ids).
- Multi-label outputs (`output.type: labels`): a set of classes per row, as separated text
  (`output.label_separator`) or lists (Parquet list columns). Set agreement, per-class
  removed/added transitions, per-class prevalence (Tango), invalid labels, exact-set accuracy
  and macro-F1 with a bootstrap over distinct row types; mean Jaccard similarity and a
  per-class table in the reports.
- Structured outputs (`output.type: structured`): one JSON object per row (text or Parquet
  struct), with `output.fields` declaring each field's type (`score`, `label`, `labels`),
  normalisation, ground truth and gates. Gates are reported as `field.gate`
  (`GateResult.field`, `field` in the JSON report). New top-level `schema_valid_rate` gate;
  invalid documents are judged once and excluded from field comparisons.

## [0.2.0] - 2026-10-02

### Added

- Class outputs: `output.type: label` (one predicted class per row) and
  `output.type: probabilities` (one probability column per class, `output.prob_prefix`).
- Class gates: `label_agreement` (Clopper-Pearson lower bound, also per segment),
  `transitions` (per class pair, Clopper-Pearson upper bound, `min_class_size`),
  `class_prevalence` (Tango score interval for paired proportions), `kappa`
  (Fleiss-Cohen-Everitt variance), `quality_difference` against `columns.truth` (accuracy via
  Tango, macro-F1 via a multinomial bootstrap over row types that scales to millions of rows).
- `tv_distance` and per-class `max_abs_diff`/`quantile_abs_diff`/`mean_diff_equivalence` for
  probability vectors; presets keep only the gates that apply to the output type.
- Report: transition-matrix heatmap in HTML, class agreement and kappa in Markdown, guidance
  when a sample is too small to verify a tolerance (rows needed per class).
- CLI: `--output-type`, `--truth`, `--prob-prefix`.
- Docs: "How many rows?" in docs/choosing-tolerances.md.

### Security

- Markdown reports neutralise backticks and line breaks in user-provided names (segments,
  classes) so they cannot inject Markdown into pull-request comments.

### Changed

- Internal: comparisons dispatch on an output kind (`output: {type: score}`, the default). No
  behaviour change for score outputs: a golden test asserts that reports match those produced
  by the released 0.1.0.

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

[Unreleased]: https://github.com/lucianoon/scoreparity/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/lucianoon/scoreparity/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lucianoon/scoreparity/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lucianoon/scoreparity/releases/tag/v0.1.0
