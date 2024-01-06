from torch.utils.data import Dataset
from typing import List, Tuple, Optional
import torch
from abc import ABC, abstractmethod


class MMDataset(Dataset):
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def __len__(self):
        pass

    @property
    def num_modalities(self):
        pass

    @property
    def has_target(self):
        pass


class SampleMMDataset(MMDataset):
    """
    Generic torch dataset object for supervised multi-modal data.
    """

    def __init__(
        self, tensors: List[torch.Tensor], target: Optional[torch.Tensor] = None
    ):
        """
        Args:
            tensors(List[torch.Tensor]): modalities for each sample, note that the first dim of each tensor is
                the sample dim
            target(torch.Tensor): label for each sample, optional to allow SSL datasets
        """
        self.tensors = tensors
        self.target = target

    def __getitem__(
        self, idx: int
    ) -> [Tuple[List[torch.Tensor], torch.Tensor], List[torch.Tensor]]:
        if self.target is None:
            return [t[idx] for t in self.tensors]
        else:
            return [t[idx] for t in self.tensors], self.target[idx]

    def __len__(self):
        return self.tensors[0].size()[0]

    @property
    def num_modalities(self):
        return len(self.tensors)

    @property
    def has_target(self):
        return False if self.target is None else True


if __name__ == "__main__":
    pass
