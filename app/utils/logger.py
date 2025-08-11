import os
import sys
import logging
from datetime import datetime
from loguru import logger
from app.config.app_config import settings

class InterceptHandler(logging.Handler):
    """
    Redirects standard logging records to Loguru.
    """
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where the logging call was made
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

def setup_logging() -> None:
    """
    Configure Loguru and intercept standard logging (uvicorn, FastAPI, etc.).
    Call this before creating your FastAPI app.
    """
    # This check prevents the logger from being configured twice when using --reload
    if "SETUP_LOGGING_COMPLETE" in os.environ:
        return
    os.environ["SETUP_LOGGING_COMPLETE"] = "True"
    
    log_level = settings.LOG_LEVEL.upper()
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"app_{run_timestamp}.log")


    # 1) Remove all existing Loguru handlers
    logger.remove()

    # 2) Add a new handler for stdout
    logger.add(
        sys.stdout,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        enqueue=True,        # thread- and process-safe
        backtrace=True,      # show backtrace on exceptions
        diagnose=True,       # variable inspection
    )

    # 3) Add a new handler for file logging
    logger.add(
        log_file_path,
        level=log_level,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} - "
            "{message}"
        ),
        enqueue=True,
        backtrace=True,
        diagnose=True,
        retention="2 days", # Automatically clean up logs older than 2 days
        compression="zip",   # Compress old log files
    )


    # 4) Intercept standard logging
    intercept = InterceptHandler()
    logging.root.handlers = [intercept]
    logging.root.setLevel(log_level)

    # 5) Make sure uvicorn/fastapi do not add their own handlers
    for pkg in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        log = logging.getLogger(pkg)
        # Clear any existing handlers and set our intercept handler
        log.handlers.clear()
        log.addHandler(intercept)
        log.propagate = False