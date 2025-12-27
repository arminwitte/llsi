Thank you for considering to contribute. Any exchange and help is welcome. However, I have to ask you to be patient with me responding.

## Development Setup

This project is optimized for use with [uv](https://github.com/astral-sh/uv).

To set up your development environment:

```bash
# Create a virtual environment and install dependencies (including dev)
uv sync --all-extras
```

Alternatively, using standard pip:

```bash
pip install -e .[dev]
```

## Code Style

This project uses `ruff` for code formatting and linting. Please ensure your code is formatted before submitting a pull request.

To check for linting errors:

```bash
uv run ruff check src/ tests/
```

To format your code:

```bash
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
```

## Testing

We use `pytest` for testing and `coverage` for code coverage analysis.

To run tests:

```bash
uv run coverage run -m pytest
```

To view the coverage report:

```bash
uv run coverage report
```

## Building and Releasing

To build the package:

```bash
uv run python -m build
```

To upload to PyPI (for maintainers):

```bash
uv run python -m twine upload --skip-existing dist/*
```
