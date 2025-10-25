import functools
import json
import os
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from dotenv import load_dotenv

from chat_agh.utils import logger

load_dotenv()
GEMINI_API_KEY = json.loads(os.environ["GEMINI_API_KEYS"])[0]

P = ParamSpec("P")
R = TypeVar("R")


def log_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        class_name = args[0].__class__.__name__ if args else func.__qualname__
        fun_name = func.__name__
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.info(f"[{class_name}.{fun_name}] Execution time: {end - start:.4f}s")
        return result

    return wrapper


def retry_on_exception(
    attempts: int = 3,
    delay: int = 1,
    backoff: int = 10,
    exception: type[Exception] = Exception,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    A decorator to retry a function call if it raises a specified exception.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            current_delay = delay
            last_exception: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exception as e:
                    last_exception = e
                    if attempt == attempts:
                        raise
                    else:
                        logger.info(
                            f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds..."
                        )
                        logger.info(
                            f"Attempt {attempt} failed: {e}. Retrying in {current_delay} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("retry_on_exception reached an unexpected state")

        return wrapper

    return decorator
