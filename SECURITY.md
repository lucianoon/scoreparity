# Security policy

## Supported versions

Only the latest released minor version receives security fixes while the project is in 0.x.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private vulnerability reporting:
<https://github.com/lucianoon/scoreparity/security/advisories/new>. You should receive an
acknowledgement within a few days; fixes are released as patch versions with an advisory.

## What this project does to stay safe

- Reports treat all data as untrusted: segment names, column names and file names are escaped
  in HTML, Markdown, JUnit XML and generated YAML (covered by tests with injection payloads).
- The GitHub Action never interpolates inputs into shell scripts (inputs travel through
  environment variables and bash arrays).
- Workflows run with least-privilege `permissions`, pin third-party actions to commit SHAs and
  do not persist checkout credentials.
- CI audits the locked dependency set for known vulnerabilities (`pip-audit`, weekly as well as
  on every change), analyses code and workflows with CodeQL and lints workflows with `zizmor`.
- Releases are built in CI, published to PyPI with Trusted Publishing (no long-lived tokens)
  and carry signed build-provenance attestations.
