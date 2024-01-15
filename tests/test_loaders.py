import torch
import pytest
from mmhb.loader import *

def test_base():
    data = MMDataset(config="config/config.yml")

    with pytest.raises(NotImplementedError):
        data[0]
    assert (len(data)) == 0
    assert data.num_modalities == 0

def test_sample_dataset():

    n = 500
    tab = torch.randn(n, 1, 50)
    img = torch.randn(n, 512, 512, 3)
    patch_img = torch.randn(n, 100, 1024)
    seq = torch.randn(n, 1, 100)
    target = torch.randn(n)

    # Generate list of tensors
    tensors = [tab, patch_img, seq]

    data = MMSampleDataset(config="config/config.yml",
                           tensors=tensors,
                           target=target
                           )

    assert len(data) == n
    assert data.num_modalities == 3


def test_tcga():
    data = TCGADataset(config="config/config.yml",
                       sources=["omic", "slides"])
    assert len(data) == 1019
    tensors = data[0]
    assert len(tensors) == 2
    pass

def test_tcga_survival():
    data = TCGASurvivalDataset(config="config/config.yml",
                               dataset="brca")
    pass