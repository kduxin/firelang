import logging
from rich.logging import RichHandler

__all__ = ['logger']

def setup_rich_logger():
    FORMAT = "%(message)s"
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
    )
    logger = logging.getLogger("rich")
    return logger

logger = setup_rich_logger()