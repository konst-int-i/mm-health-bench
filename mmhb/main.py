"""
Run all experiments
"""

from mm_lego.pipeline import Fusion
from mm_lego.utils import Config, setup_logging

logger = setup_logging()


def run():
    config = Config("config/config.yml").read()
    print(config)
    pipeline = Fusion(config=config)
    pipeline.run()


if __name__ == "__main__":
    run()
