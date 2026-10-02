# Integrations

`scoreparity` is a command with stable exit codes (0 equivalent, 1 not equivalent, 2 error) and
a versioned JSON report, so it fits any CI or orchestration system. The patterns below are the
common ones.

## GitHub Actions

```yaml
jobs:
  parity:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write     # only for the PR comment
    steps:
      - uses: actions/checkout@v7
      - run: python score.py --model ./model --out candidate.parquet
      - uses: lucianoon/scoreparity@v0.2.0
        with:
          reference: baseline/scores.parquet   # e.g. downloaded from the last release
          candidate: candidate.parquet
          config: scoreparity.yaml
```

Inputs: `reference`, `candidate`, `config` or `preset`, `context`, `args` (extra CLI flags),
`fail-on-mismatch`, `comment-on-pr`, `artifact-name`, `python-version`, `github-token`.
Outputs: `verdict`, `exit-code`, `report-dir`. For supply-chain safety you can pin the action to
a full commit SHA instead of a tag.

## GitLab CI (or any CI with JUnit support)

```yaml
score-parity:
  image: python:3.12-slim
  script:
    - pip install "scoreparity[parquet] @ git+https://github.com/lucianoon/scoreparity@v0.2.0"
    - python score.py --out candidate.parquet
    - scoreparity compare --reference baseline.parquet --candidate candidate.parquet
        --config scoreparity.yaml --junit junit.xml --html parity.html --markdown parity.md
  artifacts:
    when: always
    reports:
      junit: junit.xml        # each gate shows up as a test in the merge request
    paths: [parity.html, parity.md]
```

Jenkins (`junit 'junit.xml'`) and Azure DevOps (`PublishTestResults@2` with `testResultsFormat:
JUnit`) work the same way.

## Amazon SageMaker Pipelines

Run the comparison in a Processing step whose image has `scoreparity` installed, write the JSON
report as a step output, and let a `ConditionStep` read the verdict. The comparison step should
exit 0 even when the verdict is FAIL, so the pipeline (not the job) decides what happens next:

```python
from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.properties import PropertyFile

report = PropertyFile(name="ParityReport", output_name="report", path="report.json")
# container command (the trailing `|| true` keeps exit code 1 from failing the job;
# exit code 2, an input error, should still fail it, so test for it explicitly):
#   scoreparity compare ... --json /opt/ml/processing/output/report/report.json; \
#   code=$?; [ "$code" -le 1 ] || exit "$code"
gate = ConditionStep(
    name="ParityGate",
    conditions=[
        ConditionEquals(
            left=JsonGet(step_name=parity_step.name, property_file=report, json_path="verdict"),
            right="PASS",
        )
    ],
    if_steps=[register_model_step],  # e.g. RegisterModel with PendingManualApproval
    else_steps=[fail_step],
)
```

Attach the report to the model package (`ModelMetrics` statistics pointing at the JSON in S3) so
the registry keeps the evidence next to the version it approved.

## Airflow, Dagster, Prefect, Makefiles

Run the CLI and use the exit code: most orchestrators treat a non-zero exit as a task failure.
Distinguish "not equivalent" (1) from "could not compare" (2) when the reaction should differ
(for example, page someone only on 2).

## Re-rendering reports

The JSON report is self-sufficient. A job that only produced JSON can be turned into HTML or
JUnit later, anywhere:

```bash
scoreparity render s3-download/report.json --html parity.html --junit junit.xml
```
