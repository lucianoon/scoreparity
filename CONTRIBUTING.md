# Contributing

## Development setup

```bash
uv sync                    # creates .venv with the dev dependencies
uv run pytest              # tests
uv run ruff check . && uv run ruff format --check .
uv run mypy                # strict type checking
```

## Ground rules

- Every statistical procedure needs a test that checks its *statistical* behaviour (for
  example, the false-positive rate of an equivalence test under the null), not only that the
  code runs.
- The CLI exit codes (0 pass, 1 fail, 2 error) and the JSON report schema are public
  contracts. Changing them is a breaking change.
- Keep the core dependency set small: numpy, pandas, scipy, pyyaml.
- Update `CHANGELOG.md` under "Unreleased" in the same pull request.
