from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union
from mmhb.utils import Config
from pathlib import Path
import torch

class MMDataset(Dataset):
    """
    Generic torch dataset object for supervised multi-modal data.
    """
    def __init__(self, config: Union[str, Path]):
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

    def __getitem__(self, idx) -> [Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        # to be implemented in each child class
        pass

    def __len__(self):
        if self.tensors is None:
            return 0
        else:
            return self.tensors[0].size()[0]

    @property
    def num_modalities(self):
        return len(self.tensors)

class MMSampleDataset(MMDataset):
    pass

if __name__ == "__main__":

    pass
    # dataset = MMDataset([torch.randn(10, 3, 224, 224), torch.randn(10, 3, 224, 224)], torch.randint(0, 10, (10,)))
    # print(dataset[0])
    # print(len(dataset))
    # print(dataset.num_modalities)

