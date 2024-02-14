# Multi-modal health benchmarks
Easy access of various multi-modal biomedical datasets, ready to use in Machine Learning pipelines. 

**This is a working repository and I will keep adding resources throughout the next weeks.** 

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


## Quickstart

ChestX example
```bash
mamba env create -f environment.yml 
invoke download --dataset chestx
```

```python
from mmhb.loader import *
from mmhb.utils import Config

config = Config("config/config.yml").read()

data = ChestXDataset(data_path="data/chestx", max_seq_length=256)

(img_tensor, report_tensor), target = data[0] -> returns pytorch tensors

# can now wrap in PyTorch dataloaders...
```

## Overview

| Dataset | Download                                                  | Usage                        | Details                                                                                                                  |
|---------|-----------------------------------------------------------|------------------------------|--------------------------------------------------------------------------------------------------------------------------|
| Chest-X | `invoke download --dataset chestx`                        | `mmhb.loaders.ChestXDataset` | [Link](https://openi.nlm.nih.gov/faq) | 
| TCGA    | `invoke download --dataset tcga --sites ["brca", "luad"]` | `mmhb.loaders.TCGADataset`   | [Link](https://www.cancer.gov/ccg/research/genome-sequencing/tcga)                                                       |
| ADNI | tbd                                                       | tbd                          | [Link](http://adni.loni.usc.edu/)                                                                                         |
| PPMI | tbd                                                       | tbd                          | [Link](https://www.ppmi-info.org/)                                                                                        |
| MIMIC-IV | tbd                                                   | tbd                          | [Link](https://mimic-iv.mit.edu/)                                                                                         |




## Modality shapes

The dataloaders all return data of the following shapes

* Image: `(n h w c)`
* Image (patched): `(n p d)` # p is the number of patches, d is the flattened patch size
* Tabular: `(n d)`
* Sequences: `(n s)` # s is the number of tokens in the sequence


We offer the option to pass in `expand=True` to the multimodal dataloaders which will introduce "empty" channels if 
if you want the tensor shapes to be more consistent. This will expand the shapes to: 

* Image: `(n h w c)`
* Image (patched): `(n 1 p d)`
* Tabular: `(n 1 d)`
* Sequences: `(n 1 s)`

## Datasets supported


### Semi restricted access

The datasets in this section are open-source but may require you to register on their sites for a research project. 

#### TCGA

Collection of 14 sites from The Cancer Genome Atlas containing data on: 
- Images
  - Histopathology Slides
- Tabular
  - Gene epressions
  - Mutations
  - Copy Number Variations

The modalities are consistent across TCGA cohorts. 

Example use:
```python
from mmhb.loader import *
from mmhb.utils import Config

config = Config("config/config.yml").read()

# Task-agnostic
# This class can be used as base for a variety of possible tasks with TCGA
data = TCGADataset(
    **config.to_dict(),
    dataset="brca", 
    expand_dims=False, 
    modalities=["omic", "slides"], 
)
# access sample
omic_tensor, slide_tensor = data[0] 


# One example of this task-specific is in survival analysis
data = TCGASurvivalDataset(
    **config.to_dict(),
    dataset="brca", 
    modalities=["omic", "slides"],
)

(omic_tensor, slide_tensor), censorship, event_time, target = data[0]
```


## Setup 

Install or update the conda environment using and then activate

### Conda/Mamba

We recommend downloading [mamba](https://github.com/mamba-org/mamba) or micromamba for faster environment management. 

```
mamba env update -f environment.yml
conda activate bench
```

### Pre-commit hooks

We use `black` as a code formatting standard. To use, run: 

```bash
pre-commit install
```


### Git LFS

Some smaller preprocessed files are temporarily stored using large file storage (`git-lfs`). 
```
brew install git-lfs
git lfs install
git lfs pull
```

### Running test

```bash
pytest tests/
```