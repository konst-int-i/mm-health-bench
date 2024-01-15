import logging
from mmhb.utils.config import Config

def setup_logging():
    log_conf = Config("config/logging.yml").read()
    log_level = logging.DEBUG if log_conf.level == "debug" else logging.INFO
    logging.basicConfig(level=log_level, format=log_conf.format)
    logger = logging.getLogger(__name__)
    return logger

def check_config(config: Config):
    valid_levels = ["debug", "info"]
    assert config.level in valid_levels, f"Invalid logging level, must be one of {valid_levels}"