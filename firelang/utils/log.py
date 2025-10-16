import logging

__all__ = ['logger']

def setup_rich_logger():
    FORMAT = "%(message)s"

    try:
        from rich.logging import RichHandler
        handler = RichHandler()
    except ImportError:
        handler = logging.StreamHandler()

    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[handler]
    )
    logger = logging.getLogger("rich")
    return logger

logger = setup_rich_logger()