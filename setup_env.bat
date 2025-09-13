@echo off
echo Setting up Transformer Trading environment...

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt

REM Install development dependencies
echo Installing development dependencies...
pip install pytest-cov black isort flake8 mypy pre-commit

REM Setup pre-commit hooks
echo Setting up pre-commit hooks...
pre-commit install

REM Validate environment
echo Validating environment...
python scripts\validate_environment.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ⚠️  Environment validation failed. Some features may not work properly.
    echo Please review the validation output and address any issues.
    echo.
) else (
    echo.
    echo ✅ Environment validation passed!
    echo.
)

REM Run quick benchmark
echo Running performance benchmark...
python -c "from src.utils.benchmark_utils import run_quick_benchmark; print('Benchmark completed successfully')"

echo Environment setup complete!
echo.
echo To activate the environment, run: venv\Scripts\activate
echo To validate environment, run: python scripts\validate_environment.py
echo To run benchmark, run: python -c \"from src.utils.benchmark_utils import run_quick_benchmark\"
echo.