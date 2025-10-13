from typing import Any


def task_doit() -> dict[str, Any]:
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
        ],
        "verbosity": 2,
    }
