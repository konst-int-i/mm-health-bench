from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union
from mmhb.utils import Config
from pathlib import Path
import torch

class MMDataset(Dataset):
    """
    Generic torch dataset object for supervised multi-modal data.
    """
    def __init__(self,
                 config: Union[str, Path],
                 ):
        """
        Args:
            tensors(List[torch.Tensor]): modalities for each sample, note that the first dim of each tensor is
                the sample dim
            target(torch.Tensor): label for each sample, optional to allow SSL datasets
        """
        if type(config) is str:
            config = Path(config)
        self.config = Config(config).read()
        self.path = self.config.data_path
        self.tensors = None
        self.target = None

    def __getitem__(self, idx) -> [Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        # to be implemented in each child class
        raise NotImplementedError

    def __len__(self):
        return 0 if self.tensors is None else self.tensors[0].size()[0]

    @property
    def num_modalities(self):
        return 0 if self.tensors is None else len(self.tensors)

class MMSampleDataset(MMDataset):
    def __init__(self, config: Union[str, Path], tensors: List[torch.Tensor], target: torch.Tensor):
        super().__init__(config)
        self.tensors = tensors
        self.target = target



if __name__ == "__main__":

    pass
    # dataset = MMDataset([torch.randn(10, 3, 224, 224), torch.randn(10, 3, 224, 224)], torch.randint(0, 10, (10,)))
    # print(dataset[0])
    # print(len(dataset))
    # print(dataset.num_modalities)

