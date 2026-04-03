# Contributing

Thank you for your interest in contributing to llm-judge.

## Getting Started

```bash
uv sync
cp .env.example .env
```

## Running Tests

```bash
uv run pytest tests/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Run the full test suite and confirm all tests pass
5. Submit a pull request

## Guidelines

- Keep PRs focused — one feature or fix per PR
- Add tests for new functionality
- Follow existing code style (no linter config yet — just match what's there)

## Reporting Issues

Open an issue on GitHub. Include:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Python version and OS
