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
        # config: Union[str, Path],
        data_path: Union[str, Path],
        expand: bool = True,
        modalities: List[str] = None,
    ):
        """
        Base class for multimodal datasets. Note that each child class should implement the following methods:
        __getitem__()
        __len__()
        shapes()

        and should assign the properties from a single sample
        self.tensor # note that the tensors should be implemented in the same order as modalities are specified in self.modalities
        self.target

        Args:
            data_path (Union[str, Path]): path to data source directory
            expand (bool, optional): whether to expand dimensions across tensors with mismatching dims
        """
        self.expand = expand
        self.data_path = Path(data_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {self.device}")
        self.modalities = modalities
        self.tensors = None
        self.targets = None

        # individual sample
        self.tensor = None
        self.targets = None

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

    def shapes(self):
        shape_dict = {}
        [
            shape_dict.update({mod: t.shape})
            for t, mod in zip(self.tensor, self.modalities)
        ]
        return shape_dict

    @property
    def info(self):
        return f"{self.__class__.__name__} with {len(self)} samples and {self.num_modalities} modalities"


class MMSampleDataset(MMDataset):
    def __init__(
        self,
        # config: Union[str, Path],
        data_path: Union[str, Path],
        tensors: List[torch.Tensor],
        targets: torch.Tensor,
        **kwargs,
    ):
        super().__init__(data_path, **kwargs)
        self.tensors = tensors
        self.targets = targets

        self.tensor, _ = self.__getitem__(0)

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx: int) -> Union[List, Tuple]:
        tensors = [t[idx] for t in self.tensors]
        target = self.targets[idx]
        return (tensors, target)


if __name__ == "__main__":
    pass
    # dataset = MMDataset([torch.randn(10, 3, 224, 224), torch.randn(10, 3, 224, 224)], torch.randint(0, 10, (10,)))
    # print(dataset[0])
    # print(len(dataset))
    # print(dataset.num_modalities)
