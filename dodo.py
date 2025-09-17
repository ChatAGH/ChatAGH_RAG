def task_doit():
    """
    Run static type checks, formatting, linting,
    coverage measurement, and unit tests.
    """
    return {
        "actions": [
            # 1. Type checking
            "poetry run mypy .",
            # 2. Code linting
            "poetry run ruff check .",
            # 3. Formatting files
            "poetry run ruff check . --fix",
            # 4. Test coverage (will also run pytest)
            "poetry run coverage run -m pytest tests",
            "poetry run coverage report -m", # show uncovered files
            "poetry run coverage html",
        ],
        "verbosity": 2,
    }
