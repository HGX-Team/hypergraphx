# Contributing to Hypergraphx

Thanks for contributing to Hypergraphx.

## Development Setup

1. Fork the repository and clone your fork.
2. Create and activate a virtual environment.
3. Install the package in editable mode with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,viz]"
pre-commit install
```

## Local Quality Checks

Run these before opening a pull request:

```bash
black --check .
ruff check .
pytest
python -m build
twine check dist/*
```

If you want to run pre-commit hooks across the repository:

```bash
pre-commit run --all-files
```

## Documentation

Build docs locally when your change affects docs, APIs, or tutorials:

```bash
python -m pip install -e ".[docs]"
make -C docs html
```

## Branch and Commit Guidelines

- Create a focused branch from `main`.
- Keep pull requests small and scoped to one change.
- Write clear commit messages in imperative mood (for example: `Add temporal centrality regression test`).
- Add or update tests for behavioral changes.
- Update docs when user-facing behavior changes.

## Pull Request Checklist

- Tests added/updated for new behavior.
- Local quality checks pass.
- Docs updated (if relevant).
- PR description explains motivation, approach, and impact.

## Reporting Bugs and Requesting Features

- Use GitHub Issues for bugs and feature requests.
- Include a minimal reproducible example for bugs.
- For security issues, do not open a public issue. See `SECURITY.md`.

## Code of Conduct

By participating in this project, you agree to follow the Code of Conduct in `CODE_OF_CONDUCT.md`.
