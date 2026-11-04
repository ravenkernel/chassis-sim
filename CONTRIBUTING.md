# Contributing to chassis-sim

Thanks for taking the time to contribute. This page covers the development
environment, the gates a pull request must pass, the code conventions, and what a
reviewable pull request looks like.

- Bug reports and feature requests go through the issue forms in `.github/ISSUE_TEMPLATE/`.
- Security problems must **never** be a public issue — see `SECURITY.md`.

## Development environment

Python 3.10 or newer (CI runs the suite on 3.10, 3.11, 3.12 and 3.13). From a checkout:

```bash
git clone https://github.com/ravenkernel/chassis-sim.git
cd chassis-sim
python -m pip install --upgrade pip
pip install -e ".[dev,viz,planning]"
```

Install all three extras even when you are only touching the core. Parts of the suite
call `pytest.importorskip` on `matplotlib` and `numba`, so a core-only checkout makes
those tests skip rather than run — green, but not the same green. CI keeps a separate
`core-only` job (`pip install -e ".[dev]"`) precisely so that skipping stays a property
of that job and not of yours.

The `dev` extra carries the tools the gates need: `mypy`, `pytest-cov` and
`types-PyYAML` at pinned versions, plus `ruff` (`>=0.4` is a floor, not a pin — the
`lint` job pins the exact `0.17.3` it lints with). That makes
`pip install -e ".[dev,viz,planning]"` the whole local setup — CI installs the same
extras and adds nothing on top, so the `types` and `coverage` jobs cannot be running a
different version than your checkout. Those pins live in `pyproject.toml`; bumping one
is a single edit and belongs in its own pull request.

## The gates

Five gates plus the coverage floor: run all six from the repository root before you
push, and each must exit 0. This table is a line-by-line mapping onto
`.github/workflows/ci.yml`:

| Gate | Local command | `.github/workflows/ci.yml` |
| --- | --- | --- |
| tests | `env -u PYTHONPATH pytest -q` | job `test` → step `Test` → `pytest -q` (matrix 3.10 / 3.11 / 3.12 / 3.13 × `numpy<2` / `numpy>=2`; 3.13 has no `numpy<2` cell) |
| docs | `env -u PYTHONPATH python tools/check_readme_blocks.py` | job `docs` → step `Run README python blocks` → `python tools/check_readme_blocks.py` |
| lint | `ruff check .` | job `lint` → step `ruff check` → `ruff check .` (ruff 0.17.3) |
| format | `ruff format --check .` | job `lint` → step `ruff format --check` → `ruff format --check .` |
| types | `MYPYPATH=src mypy` | job `types` → step `mypy` → `env: MYPYPATH: src`, run `python -m mypy` |
| coverage | `env -u PYTHONPATH pytest -q --cov --cov-report=term:skip-covered --cov-fail-under=82` | job `coverage` → step `Test with coverage gate` → the same command |

That `tests` gate is the same command on every runner: CI runs the suite on ubuntu
(Python 3.10–3.13) and on macOS / Windows (Python 3.12, job `test-os`), so a
platform-specific failure lands in the gate this pull request is measured against
rather than in the first run of a user.

Why the local commands differ, where they do:

- `env -u PYTHONPATH` is the only local-only prefix. CI starts without a `PYTHONPATH`,
  while a shell that has one set can shadow the editable install and import a stale
  copy of the package. Stripping it locally is what makes `pytest -q` mean the same
  thing in both places.
- `mypy` and `python -m mypy` are the same run: `[tool.mypy] files = [...]` in
  `pyproject.toml` fixes the file list, so neither call takes a path argument.
  Extending type coverage means editing that list, not the workflow.
- The `82` appears on the coverage command on purpose. The authoritative value is
  `[tool.coverage.report] fail_under = 82` in `pyproject.toml`; the copy on the command
  line is what puts the threshold in a failing PR log. Change both or neither.
- `ruff format --check .` only reports. When it fails, format the specific files you
  touched (`ruff format path/to/file.py`) and read the resulting diff — do not rewrite
  the whole tree to make a gate quiet.
- `tools/check_readme_blocks.py` executes every `python` block in `README.md` and
  `README.zh-CN.md` in its own subprocess, in its own temporary directory, so a block
  that writes a file cannot leave it in your checkout. A block that cannot run in CI
  starts with `# no-exec` (optionally `# no-exec: 需要 GPU`, i.e. with the reason) —
  that counts as *skipped* and is printed on every run. A skip is not a pass: the
  reason is what makes it reviewable. The script exits 2 when a README has no
  `python` block at all, when a fence is unclosed, or when the file is missing, so a
  rewrite that drops the blocks fails this gate instead of passing vacuously.

`.pre-commit-config.yaml` wires the same two ruff checks plus whitespace / YAML / TOML
hygiene. It is a convenience, not a gate: `pre-commit run --all-files` is optional, and
the five commands above stay authoritative.

## Code conventions

- **Comments and error messages are Chinese.** That is the existing convention
  throughout `src/`, `tests/` and `pyproject.toml`, and new code follows it — including
  the text of `raise ValueError("...")`. Public identifiers, docstrings aimed at users
  and the READMEs stay English (+ 简体中文), matching the files around them. Commit
  subjects follow the existing `type(scope): 中文说明` style.
- **Do not lower the coverage floor.** `fail_under = 82` is measured, not aspirational.
  New code arrives with tests; a drop is a bug in the pull request, not in the gate.
- **Do not widen the linter's blind spots.** The selected ruff families, the unselected
  families and their measured finding counts are documented inside `pyproject.toml`.
  Fix a new finding instead of adding `noqa` or a per-file ignore; if you believe a
  rule is wrong for this codebase, open an issue with the argument.
- **Do not append to the mypy `[[tool.mypy.overrides]]` list.** Each entry there is a
  documented historical exception with a comment naming the exact line; a new entry
  hides a new error. Whoever fixes a listed one deletes its entry in the same PR.
- **Keep the two READMEs in sync.** `README.md` and `README.zh-CN.md` share their
  section structure and carry the same number of `python` blocks, with byte-identical
  block contents; a change to one is a change to both. Both files have to stay
  executable: `python tools/check_readme_blocks.py` runs every block, and `# no-exec`
  with a reason is the only way to skip one.
- **No version numbers in prose.** `pyproject.toml` is the single source of truth for
  the version, and the READMEs describe the current line rather than a specific
  release. `3.0.0` is the API-stability line: from it on the package follows semver,
  which is why the classifier stays `Development Status :: 5 - Production/Stable`
  (the sentence is at the top of `CHANGELOG.md`).
  Version bumps and `CHANGELOG.md` entries are the maintainer's job, in the release
  commit — please leave them out of feature PRs.

## Pull requests

- **Small and focused.** One concern per pull request. A fix mixed with a whitespace
  cleanup or an unrelated rename will be sent back, because it cannot be reviewed or
  reverted as a unit.
- **With tests.** New behavior gets a test that fails without it; a bug fix gets a test
  reproducing the bug. A new public function without a test is treated as untested
  behavior, not as a feature.
- **Gates green, not "nearly".** All six commands above run locally before the PR is
  opened. A red gate that CI turns green because a test skipped is not green — the
  same holds for a README block that was made to skip instead of to pass.
- **Docs for user-visible changes.** A changed default, a renamed parameter or a new
  extra belongs in both READMEs, in the section each one already has for it.

## Before you open a pull request

- [ ] `env -u PYTHONPATH pytest -q` passes.
- [ ] `ruff check .` passes.
- [ ] `ruff format --check .` passes.
- [ ] `MYPYPATH=src mypy` passes.
- [ ] `env -u PYTHONPATH pytest -q --cov --cov-report=term:skip-covered --cov-fail-under=82` passes (still at or above 82%).
- [ ] `env -u PYTHONPATH python tools/check_readme_blocks.py` passes (0 failed; every
      skip carries a `# no-exec` reason).
- [ ] New behavior has a test; a bug fix has a test that fails without the fix.
- [ ] Both READMEs updated if user-visible behavior changed, with byte-identical `python` blocks.
- [ ] Comments and error messages are Chinese, matching the surrounding file.
