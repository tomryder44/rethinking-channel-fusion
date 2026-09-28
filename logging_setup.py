import logging


def setup_logging(log_path: str = "log.log") -> None:
    logging.basicConfig(
        filename=log_path,
        filemode="w",  # a - append, w - overwrite
        level=logging.DEBUG,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%m-%d %H:%M:%S",
        force=True,
    )
