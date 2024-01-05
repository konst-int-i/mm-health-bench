# Multi-modal health benchmarks
Easy access of various multi-modal healthcare datasets for Machine Learning pipelines. 

## Quickstart

tbd

## Modality shapes

The dataloaders all return data of the following shapes

* Image: `(n h w c)`
* Image (patched): `(n p d)` # p is the number of patches, d is the flattened patch size
* Tabular: `(n d)`
* Sequences: `(n s)` # s is the number of tokens in the sequence


We offer the option to pass in `expan=True` to the multimodal dataloaders which will introduce "empty" channels if 
if you want the tensor shapes to be consistent. This will be matched to the shape of the highest-dimensional modality. 

* Image: `(n h w c)`
* Image (patched): `(n 1 p d)`
* Tabular: `(n 1 1 d)`
* Sequences: `(n 1 1 s)`

## Datasets supported

Pathology: 
* TCGA (14 cancer sites)
* Camelyon17
* Camelyon16


Radiology: 


## Setup 

Install or update the conda environment using and then activate

### Conda/Mamba

We recommend downloading [mamba](https://github.com/mamba-org/mamba) or micromamba for faster environment management. 

```
mamba env update -f environment.yml
conda activate bench
```

### Git LFS

Some smaller preprocessed files are temporarily stored using large file storage (`git-lfs`). 
```
brew install git-lfs
git lfs install
git lfs pull
```
