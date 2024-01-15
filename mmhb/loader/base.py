from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union
from mmhb.utils import Config, setup_logging
import logging
from pathlib import Path
import torch

logger = setup_logging()


class MMDataset(Dataset):
    """
    Generic torch dataset object for supervised multi-modal data.
    """

    def __init__(
        self,
        config: Union[str, Path],
        expand_dims: bool = True,
    ):
        """
        Args:
            config (Union[str, Path]): path to config file
            expand_dims (bool, optional): whether to expand dimensions across tensors with mismatching dims
        """
        if type(config) is str:
            config = Path(config)
        self.config = Config(config).read()
        self.expand_dims = expand_dims
        self.data_path = Path(self.config.data_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {self.device}")
        self.tensors = None
        self.target = None

    def __getitem__(
        self, idx
    ) -> [Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        # to be implemented in each child class
        raise NotImplementedError

    def __len__(self):
        try:
            sample = self.__getitem__(0)
        except NotImplementedError:
            return 0

    @property
    def num_modalities(self):
        return 0 if self.tensors is None else len(self.tensors)


class MMSampleDataset(MMDataset):
    def __init__(
        self,
        config: Union[str, Path],
        tensors: List[torch.Tensor],
        target: torch.Tensor,
    ):
        super().__init__(config)
        self.tensors = tensors
        self.target = target

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx: int) -> Union[List, Tuple]:
        tensors = [t[idx] for t in self.tensors]
        target = self.target[idx]
        return (tensors, target)


if __name__ == "__main__":
    pass
    # dataset = MMDataset([torch.randn(10, 3, 224, 224), torch.randn(10, 3, 224, 224)], torch.randint(0, 10, (10,)))
    # print(dataset[0])
    # print(len(dataset))
    # print(dataset.num_modalities)
