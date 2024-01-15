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

    data = MMSampleDataset(config="config/config.yml", tensors=tensors, target=target)

    assert len(data) == n
    assert data.num_modalities == 3


def test_tcga():
    data = TCGADataset(
        config="config/config.yml", dataset="brca", sources=["omic", "slides"]
    )
    assert len(data) == 1019
    tensors = data[0]
    assert len(tensors) == 2
    assert (data.num_modalities) == 2


def test_tcga_survival():
    n_patches = 1081  # number of image patches in first sample (data[0])
    patch_dim = 2048
    n_feats = 2914  # tabular features

    # TEST CASE - smoke test regular dims, expand_dims=False
    data = TCGASurvivalDataset(
        config="config/config.yml",
        dataset="brca",
        expand_dims=False,
        sources=["omic", "slides"],
    )
    assert data.num_modalities == 2
    assert len(data) == 1019
    tensors, censorship, event_time, target = data[0]
    # check omic shape
    assert tensors[0].shape == torch.Size([n_feats])
    # check image shape
    assert tensors[1].shape == torch.Size([n_patches, patch_dim])

    # TEST CASE - expand_dims=True
    data = TCGASurvivalDataset(
        config="config/config.yml", dataset="brca", sources=["omic"], expand_dims=True
    )
    tensors, censorship, event_time, target = data[0]
    assert tensors[0].shape == torch.Size([1, n_feats])

    # TEST CASE - concatenated dims
    data = TCGASurvivalDataset(
        config="config/config.yml",
        dataset="brca",
        sources=["omic", "slides"],
        expand_dims=False,
        concat=True,
    )
    tensors, censorship, event_time, target = data[0]
    assert len(tensors) == 1
    # expected shape img(p * d) + tab(d)
    assert tensors[0].shape == torch.Size([(n_patches * patch_dim) + n_feats])
