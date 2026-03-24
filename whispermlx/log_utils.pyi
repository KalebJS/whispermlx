import logging

def setup_logging(
    level: str = ...,
    log_file: str | None = ...,
) -> None: ...
def get_logger(name: str) -> logging.Logger: ...
