import pytest
from mmhb.utils import *


def test_config():
    # config smoke test
    config = Config("config/config.yml").read()

    assert hasattr(config, "data_path")
