<!-- Sync Impact Report
Version change: N/A → 0.1.0
Added Principles: Library-First, Type Safety, Testing Discipline, Code Quality, Documentation
Added Sections: Additional Constraints, Development Workflow, Governance
Templates updated: ✅ .specify/templates/plan-template.md, ✅ .specify/templates/spec-template.md, ✅ .specify/templates/tasks-template.md, ✅ .specify/templates/commands/*.md
TODOs: RATIFICATION_DATE unknown
-->
# PowerFit Constitution
<!-- Example: Spec Constitution, TaskFlow Constitution, etc. -->

## Core Principles

### Library-First
<!-- Example: I. Library-First -->
Every feature must be implemented as a self‑contained, reusable library. Libraries are required to be independently testable, documented, and have a clear purpose; organizational‑only code is prohibited.
<!-- Example: Every feature starts as a standalone library; Libraries must be self-contained, independently testable, documented; Clear purpose required - no organizational-only libraries -->

### Type Safety
<!-- Example: II. Type Safety -->
All public APIs MUST be fully annotated with type hints. The project MUST enforce type checking using `mypy` (or equivalent) in CI. Types are considered part of the contract and must not be removed without a major version bump.
<!-- Example: II. CLI Interface -->

<!-- Example: Every library exposes functionality via CLI; Text in/out protocol: stdin/args → stdout, errors → stderr; Support JSON + human-readable formats -->

### Testing Discipline
<!-- Example: III. Test-First (NON-NEGOTIABLE) -->
All new code MUST include unit tests covering core functionality. Tests are run via `pytest` in CI, and code coverage must be ≥80%. Test files reside in `tests/` and follow the project's naming conventions.
<!-- Example: TDD mandatory: Tests written → User approved → Tests fail → Then implement; Red-Green-Refactor cycle strictly enforced -->

### Code Quality
<!-- Example: IV. Code Quality -->
All code MUST pass the project's linting and formatting checks. The project uses `ruff` for linting and auto‑formatting; CI runs `ruff check --fix` and `ruff format --check`. Issues must be resolved before merging.
<!-- Example: IV. Integration Testing -->
All code MUST pass linting and formatting checks using `ruff`. CI runs `ruff check --fix` and `ruff format --check`. Issues must be resolved before merging.
<!-- Example: Focus areas requiring integration tests: New library contract tests, Contract changes, Inter-service communication, Shared schemas -->

### Documentation
<!-- Example: V. Observability, VI. Versioning & Breaking Changes, VII. Simplicity -->
All public modules and functions MUST have comprehensive docstrings. The README and CONTRIBUTING guidelines provide usage and contribution instructions.
<!-- Example: Text I/O ensures debuggability; Structured logging required; Or: MAJOR.MINOR.BUILD format; Or: Start simple, YAGNI principles -->

## Additional Constraints
<!-- Example: Additional Constraints, Security Requirements, Performance Standards, etc. -->

- Python >=3.10
- Dependencies declared in `pyproject.toml` must be kept up‑to‑date.
- CI must run tests, linting, type checking, and documentation build on each push.
- Release process follows `CONTRIBUTING.md` guidelines, including version bump and changelog update.
<!-- Example: Technology stack requirements, compliance standards, deployment policies, etc. -->

## Development Workflow
<!-- Example: Development Workflow, Review Process, Quality Gates, etc. -->

- Code must be reviewed via pull request before merge.
- All new features require unit tests with ≥80% coverage.
- CI runs linting, type checking, tests, and documentation build.
- Releases follow the steps in `CONTRIBUTING.md`: version bump, changelog update, tag, and publish to PyPI and Docker.
- Documentation updates must be included in the same PR as code changes.
<!-- Example: Code review requirements, testing gates, deployment approval process, etc. -->

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->


All contributors must adhere to the processes defined in `CONTRIBUTING.md`. Amendments to this constitution require a proposal, review, and approval via a merged pull request. Version increments follow semantic versioning: MAJOR for breaking changes to principles, MINOR for new principles or sections, PATCH for clarifications.
- **Amendment Procedure**: Submit a PR updating this document, obtain at least two approvals, and merge.
- **Versioning Policy**: See above.
- **Compliance Review**: CI enforces linting, type checking, testing, and documentation generation. Any PR failing these checks cannot be merged.


**Version**: 0.1.0 | **Ratified**: TODO(RATIFICATION_DATE) | **Last Amended**: 2025-11-25
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->
