# Development Guidelines

## Environment Setup

1. **Virtual Environment**: Always work within the virtual environment
   - Activate: `venv\Scripts\activate` (Windows)
   - Deactivate: `deactivate`

2. **Dependencies**: Use `requirements.txt` for all dependencies
   - Install: `pip install -r requirements.txt`
   - Update: `pip freeze > requirements.txt`

## Code Quality

### Formatting
- Use `ruff` for linting and formatting
- Run: `ruff check .` to lint
- Run: `ruff format .` to format

### Testing
- Place tests in `tests/` directory
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- E2E tests: `tests/e2e/`
- Run tests: `pytest tests/`
- Run with coverage: `pytest --cov=src tests/`

### Type Checking
- Use `mypy` for static type checking
- Run: `mypy src/`

## Project Structure

```
AlgoTrading/
├── src/                    # Source code
│   ├── models/            # ML models
│   ├── data/              # Data processing
│   ├── api/               # API endpoints
│   └── utils/             # Utilities
├── tests/                 # Test files
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── data/                  # Data files
│   ├── raw/               # Raw data
│   ├── final/             # Processed data
│   └── test/              # Test data
├── notebooks/             # Jupyter notebooks
├── config/                # Configuration files
├── logs/                  # Log files
└── docs/                  # Documentation
```

## Git Workflow

1. Create feature branches from `main`
2. Commit regularly with descriptive messages
3. Pull before pushing
4. Use conventional commit format: `type(scope): description`

## Configuration

- Environment variables: Create `.env` file (not committed)
- Config files: Place in `config/` directory
- Use `omegaconf` for configuration management