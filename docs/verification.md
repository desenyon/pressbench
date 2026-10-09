# Verification evidence for the 1.2 runner upgrade

Base inspected: `b3826a0753e85ac11da67b2626ef42c6c34a97f7` (Architecture Atlas documentation).
The preceding implementation/test commits and all tracked source were reviewed.
No repository `AGENTS.md` or `.agents/skills` instructions were present.

## Baseline

An untouched archive of the base revision passed **41 tests** on Python 3.13.12.
Static inspection found the default evaluation dataset path pointed outside the
package, JSON corpus files were not declared as package data, runs above three
conflicted with the instance schema, all tasks were scheduled up front, and
results were written only after all model calls. Failures and initially wrong
answers shared an exclusion count; beneficial corrections were counted only
inside the initially correct subset, making their rate zero.

## Final local checks

- Python 3.13.12 on macOS; isolated project environment.
- `python -m pytest -q`: **107 passed**.
- `ruff check press tests scripts`: passed.
- `ruff format --check press tests scripts`: passed, 36 Python files.
- `mypy`: passed, 22 source files.
- `git diff --check`: passed.
- `python -m build --no-isolation`: source distribution and wheel built successfully.
- Wheel installed with its dependencies in a second virtual environment under `/tmp`.
- `scripts/smoke_wheel.py` executed from `/tmp`: imported the installed package,
  loaded/validated all 500 questions, and ran the dry-run CLI successfully.
- `uv pip check` in that wheel environment: all 53 installed packages compatible.
- Wheel contains all six question JSON files; source distribution includes the
  design notes, contribution/citation metadata and smoke script.

The full CLI integration test generates actual PNG charts, JSON exports and embedded
HTML, returns nonzero for simulated partial failure, resumes using saved initial
responses, then regenerates reports from the saved aggregate. Additional tests exercise
bounded concurrency, cancellation cleanup, deadlines, incompatible and corrupt
checkpoints, atomic export failure, wrong-to-correct denominators, decimal/token
matching, and OpenAI SDK parsing/retry with a mock HTTP transport.

## Remote and unrun checks

The added CI workflow runs tests, lint, formatting, typing, build and installed-wheel
smoke on Python 3.10 and 3.13. Exact pushed-commit run status is reported in the task
handoff and can be inspected in the repository's Actions tab.

No paid live-model benchmark was run. Offline provider contracts do not establish
current model availability, API compatibility for every listed model, empirical
confidence calibration, or new model performance results. Local Python 3.10 and
Windows execution were not run; Python 3.10 is covered by the remote CI matrix when
that run succeeds.
