import logging
import sys

from loguru import logger


class InterceptHandler(logging.Handler):
    """
    Route standard-library logging records to Loguru.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    log_level: str = "INFO", enable_stdout: bool = True
) -> None:
    """
    Configure Loguru to write to stdout and intercept stdlib logging.
    """
    logger.remove()

    if enable_stdout:
        logger.add(sys.stdout, level=log_level, enqueue=True)

    intercept_handler = InterceptHandler()
    logging.basicConfig(handlers=[intercept_handler], level=log_level, force=True)

    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [intercept_handler]
        uvicorn_logger.propagate = False
