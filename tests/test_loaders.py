import pytest
import einops
import torch
from mmhb.loader import MMDataset


# @pytest.fixture(autouse=True)
def test_base_loader():
    n = 10
    dataset = MMDataset(
        tensors=[torch.randn(n, 3, 224, 224), torch.randn(n, 3, 224, 224)],
        target=torch.randint(0, 100, (n,)),
    )

    ssl_dataset = MMDataset(
        tensors=[
            torch.randn(n, 3, 224, 224),
            torch.randn(n, 3, 224, 224),
            torch.randn(n, 3, 224, 224),
        ],
    )

    assert len(dataset) == 10
    assert dataset.num_modalities == 2
    assert dataset.has_target == True
    assert ssl_dataset.num_modalities == 3
    assert ssl_dataset.has_target == False
