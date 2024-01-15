from mmhb.loader import MMDataset
from mmhb.utils import setup_logging
from typing import Union, List
from pathlib import Path

logger = setup_logging()

class TCGADataset(MMDataset):
    def __init__(self,
                 config: Union[str, Path],
                 dataset: str,
                 sources: List = ["omic", "slides"],
                 level: int=2,
                 ):
        super().__init__(config)







class TCGASurvivalDataset(TCGADataset):
    """
    Task-specific dataset
    """
    pass


if __name__ == "__main__":
    data = TCGASurvivalDataset(config="config/config.yml",
                               dataset="brca")
