import torch
import pytest
from mmhb.loader import *
from mmhb.utils import Config


@pytest.fixture(scope="module")
def config():
    return Config("config/config.yml").read()


def test_base(config):
    data = MMDataset(**config.to_dict())

    with pytest.raises(NotImplementedError):
        data[0]
    assert (len(data)) == 0
    assert data.num_modalities == 0


def test_sample_dataset(config):
    n = 500
    tab = torch.randn(n, 1, 50)
    img = torch.randn(n, 512, 512, 3)
    patch_img = torch.randn(n, 100, 1024)
    seq = torch.randn(n, 1, 100)
    target = torch.randn(n)

    # Generate list of tensors
    tensors = [tab, patch_img, seq]

    data = MMSampleDataset(**config.to_dict(), tensors=tensors, targets=target)

    assert len(data) == n
    assert data.num_modalities == 3


def test_tcga(config):
    data = TCGADataset(
        **config.to_dict(), dataset="brca", modalities=["omic", "slides"]
    )
    assert len(data) == 1019
    tensors = data[0]
    assert len(tensors) == 2
    assert (data.num_modalities) == 2


def test_tcga_survival(config):
    n_patches = 1081  # number of image patches in first sample (data[0])
    patch_dim = 2048
    n_feats = 2914  # tabular features

    # TEST CASE - smoke test regular dims, expand=False
    data = TCGASurvivalDataset(
        **config.to_dict(),
        dataset="brca",
        expand=False,
        modalities=["omic", "slides"],
    )
    assert data.num_modalities == 2
    assert len(data) == 1019
    tensors, censorship, event_time, target = data[0]
    # check omic shape
    assert tensors[0].shape == torch.Size([n_feats])
    # check image shape
    assert tensors[1].shape == torch.Size([n_patches, patch_dim])

    # TEST CASE - expand=True
    data = TCGASurvivalDataset(
        **config.to_dict(), dataset="brca", modalities=["omic"], expand=True
    )
    tensors, censorship, event_time, target = data[0]
    assert tensors[0].shape == torch.Size([1, n_feats])

    # TEST CASE - concatenated dims
    data = TCGASurvivalDataset(
        **config.to_dict(),
        dataset="brca",
        modalities=["omic", "slides"],
        expand=False,
        concat=True,
    )
    tensors, censorship, event_time, target = data[0]
    assert len(tensors) == 1
    # expected shape img(p * d) + tab(d)
    assert tensors[0].shape == torch.Size([(n_patches * patch_dim) + n_feats])


def test_chestx(config):
    data = ChestXDataset(
        data_path="data/chestx",
        max_seq_length=256,
        modalities=["images", "reports"],
        expand=True,
    )
    tensors, target = data[1]
    print(len(data))
    print(tensors[0].shape, tensors[1].shape)
    assert len(tensors) == 2
    # expecting h w c
    shapes = data.shapes()
    assert shapes["images"] == torch.Size([256, 256, 3])
    # note that we currently pad the tensors here
    assert shapes["reports"] == torch.Size([1, 256])
    print(data.shapes())
