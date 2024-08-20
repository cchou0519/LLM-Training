import logging

_logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
_logger.addHandler(_handler)
