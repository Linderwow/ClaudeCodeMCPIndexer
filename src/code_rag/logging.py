from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure(log_dir: Path, level: str = "INFO", json_to_file: bool = True) -> None:
    """Configure structlog + stdlib logging.

    Human-readable to stderr (so MCP stdio transport stays clean on stdout),
    JSON to a rolling file for post-hoc analysis.

    Under `pythonw.exe` (used by the windowless autostart), sys.stderr is None
    because there's no attached console. We substitute the file handler as the
    structlog sink in that case so logging never crashes.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "code_rag.jsonl"

    handlers: list[logging.Handler] = []
    # Only install a stderr handler if stderr actually exists.
    if sys.stderr is not None:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(stderr_handler)

    file_handler: logging.Handler | None = None
    if json_to_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        handlers.append(file_handler)

    # basicConfig requires at least one handler.
    if not handlers:
        handlers.append(logging.NullHandler())
    logging.basicConfig(level=level, handlers=handlers, force=True)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # When stderr is unavailable (pythonw), log to the JSON file instead.
    sink = sys.stderr
    colors = False
    if sink is None:
        sink = open(log_file, "a", encoding="utf-8")  # noqa: SIM115
    else:
        colors = sink.isatty()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.dev.ConsoleRenderer(colors=colors),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level)),
        logger_factory=structlog.PrintLoggerFactory(file=sink),
        cache_logger_on_first_use=True,
    )


def get(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
