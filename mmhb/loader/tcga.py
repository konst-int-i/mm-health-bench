import h5py
import pandas as pd
import threading
import tables
import einops
from openslide import OpenSlide
from torchvision import models as models, transforms
from tqdm import tqdm
import numpy as np
from mmhb.loader import MMDataset
from mmhb.utils import setup_logging, RepeatTransform, RearrangeTransform, Config
import torch
import os
from typing import Union, List, Tuple
from pathlib import Path
from tiatoolbox.tools import stainnorm
from tiatoolbox.models.engine.patch_predictor import PatchPredictor
from tiatoolbox import data

logger = setup_logging()


class TCGADataset(MMDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        dataset: str,
        modalities: List = ["omic", "slides"],
        expand: bool = False,
        level: int = 2,
        filter_overlap: bool = True,
        patch_wsi: bool = True,
        concat: bool = False,
        **kwargs,
    ):
        super().__init__(data_path, expand, modalities, **kwargs)
        self.prep_path = self.data_path.joinpath(
            f"tcga/wsi/{dataset}_preprocessed_level{level}"
        )  # preprocessed data path
        self.dataset = dataset
        self.level = level
        self.modalities = modalities
        self.filter_overlap = filter_overlap
        self.patch_wsi = patch_wsi
        self.concat = concat  # whether to flatten tensor for early fusion

        self._check_args()
        # pre-fetch data
        self.omic_df = self.load_omic()
        self.slide_ids = self.omic_df["slide_id"].str.strip(".svs")
        self.omic_df = self.omic_df.drop(
            ["site", "oncotree_code", "case_id", "slide_id", "train"], axis=1
        )
        self.omic_tensor = torch.Tensor(self.omic_df.values).squeeze()

        # required for shape() function
        self.tensor = TCGADataset.__getitem__(self, 0)

    def _check_args(self):
        assert len(self.modalities) > 0, "No sources specified"

        valid_sources = ["omic", "slides"]
        assert all(
            source in valid_sources for source in self.modalities
        ), "Invalid source specified"

        valid_datasets = [
            "blca",
            "brca",
            "kirp",
            "ucec",
            "hnsc",
            "paad",
            "luad",
            "lusc",
        ]
        assert (
            self.dataset in valid_datasets
        ), f"Invalid dataset, must be one of {valid_datasets}"

    def __getitem__(self, idx: int) -> Tuple:
        tensors = []
        if "omic" in self.modalities:
            tensor = self.omic_tensor[idx]
            if self.expand:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        if "slides" in self.modalities:
            if self.patch_wsi:
                tensor = self.load_patches(slide_id=self.slide_ids[idx])
                tensors.append(tensor)
            else:
                raise NotImplementedError("Raw WSI loader not implemented")

        if self.concat:
            tensors = torch.cat([torch.flatten(t) for t in tensors], dim=0)
            # if self.expand:
            #     # expand again for healnet
            #     tensors = tensors.unsqueeze(0)
            # # return as list (even if concatenated)
            tensors = [tensors]
        assert isinstance(tensors, list), "tensors must be a list"

        return tensors

    def __len__(self) -> int:
        return self.omic_tensor.shape[0]

    @property
    def num_modalities(self) -> int:
        return len(self.modalities)

    def load_omic(self) -> pd.DataFrame:
        load_path = self.data_path.joinpath(
            f"tcga/omic/tcga_{self.dataset}_all_clean.csv.zip"
        )
        df = pd.read_csv(
            load_path, compression="zip", header=0, index_col=0, low_memory=False
        )

        # handle missing
        num_nans = df.isna().sum().sum()
        nan_counts = df.isna().sum()[df.isna().sum() > 0]
        logger.debug(f"Filled {num_nans} missing values with mean")
        if num_nans > 0:
            df = df.fillna(df.mean(numeric_only=True))
            logger.debug(f"Missing values per feature: \n {nan_counts}")

        if self.filter_overlap:
            df = self._filter_overlap(df)

        return df

    def load_patches(self, slide_id: str) -> torch.Tensor:
        """
        Loads patch features for a single slide from torch.pt file
        Args:
            slide_id (str): Slide ID

        Returns:
            torch.Tensor: Patch features
        """
        load_path = self.prep_path.joinpath(f"patch_features/{slide_id}.pt")
        with open(load_path, "rb") as file:
            patch_features = torch.load(
                file, weights_only=True, map_location=torch.device("cpu")
            )
        return patch_features

    def _filter_overlap(self, df: pd.DataFrame):
        # slides_available = self.slide_ids
        slides_available = [
            slide_id.rsplit(".", 1)[0]
            for slide_id in os.listdir(self.prep_path.joinpath("patches"))
        ]
        omic_available = [id[:-4] for id in df["slide_id"]]
        overlap = set(slides_available) & set(omic_available)
        logger.info(f"Slides available: {len(slides_available)}")
        logger.info(f"Omic available: {len(omic_available)}")
        logger.info(f"Overlap: {len(overlap)}")
        if len(slides_available) < len(omic_available):
            logger.debug(
                f"Filtering out {len(omic_available) - len(slides_available)} samples for which there is no omic data available"
            )
            overlap_filter = [id + ".svs" for id in overlap]
            df = df[df["slide_id"].isin(overlap_filter)]
        elif len(slides_available) > len(omic_available):
            logger.debug(
                f"Filtering out {len(slides_available) - len(omic_available)} samples for which there are no slides available"
            )
            # self.slide_ids = overlap
        else:
            logger.info("100% modality overlap, no samples filtered out")

        # reset index
        df = df.reset_index(drop=True)
        return df


class TCGASurvivalDataset(TCGADataset):
    """
    Task-specific dataset
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        dataset: str,
        expand: bool = False,
        modalities: List = ["omic", "slides"],
        level: int = 2,
        filter_overlap: bool = True,
        patch_wsi: bool = True,
        n_bins: int = 4,
        **kwargs,
    ):
        super().__init__(
            data_path,
            dataset,
            modalities,
            expand,
            level,
            filter_overlap,
            patch_wsi,
            **kwargs,
        )
        self.n_bins = n_bins

        # calculate survival
        self.omic_df = self._calc_survival()
        # drop vars to avoid leakage
        self.features = self.omic_df.drop(
            ["censorship", "survival_months", "y_disc"], axis=1
        )
        self.omic_tensor = torch.Tensor(self.features.values).squeeze()

        self.censorship = self.omic_df["censorship"]
        self.event_time = self.omic_df["survival_months"]
        self.targets = torch.Tensor(
            self.omic_df["y_disc"].values
        ).long()  # cast as int64 since sksurv requires this for c-index

        # self.tensor defined in parent
        self.target = self.targets[0]

    def __getitem__(self, idx: int):
        # get list of tensors
        tensors = super().__getitem__(idx)
        return tensors, self.censorship[idx], self.event_time[idx], self.targets[idx]

    def _calc_survival(self, eps: float = 1e-6):
        survival = "survival_months"
        df = self.omic_df

        # take q_bins from uncensored patients
        subset_df = df[df["censorship"] == 0]
        disc_labels, q_bins = pd.qcut(
            subset_df[survival], q=self.n_bins, retbins=True, labels=False
        )
        q_bins[-1] = df[survival].max() + eps
        q_bins[0] = df[survival].min() - eps
        df["y_disc"] = pd.cut(
            df[survival],
            bins=q_bins,
            retbins=False,
            labels=False,
            right=False,
            include_lowest=True,
        ).values
        return df


def encode_patches(
    level: int, prep_path: Path, site_path: Path, pretraining: str = "kather"
):
    # print("slide ids", len(slide_ids))
    print(f"Prep path: {prep_path}")
    print(f"Reading from: {prep_path.joinpath('patches')}")
    slide_files = os.listdir(prep_path.joinpath("patches"))

    # load patch coords
    coords = {}
    for slide_file in slide_files:
        patch_path = prep_path.joinpath(f"patches/{slide_file}")
        file = tables.open_file(patch_path, "r")
        try:
            patch_coords = file.get_node("/coords")[:]
            slide_id = str(slide_file)[:-3]
            coords[slide_id] = patch_coords
            file.close()
        except FileNotFoundError as e:
            print(f"No patches available for file {patch_path}")
            pass
    max_patches = max([coords.get(key).shape[0] for key in coords.keys()])
    print(f"Max patches: {max_patches}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if pretraining == "imagenet":
        # load in resnet50 model
        feat_path = prep_path.joinpath("patch_features")

        patch_encoder = models.resnet50(
            weights=models.resnet.ResNet50_Weights.IMAGENET1K_V2
        )

        patch_tensors = torch.zeros(max_patches, 2048)
        transform = transforms.Compose(
            [
                transforms.Lambda(
                    lambda x: x.convert("RGB")
                ),  # need to convert to RGB for ResNet encoding
                transforms.ToTensor(),
                transforms.Resize((224, 224)),  # resize in line with ResNet50
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        encode = torch.nn.Sequential(*(list(patch_encoder.children())[:-1])).to(
            device
        )  # remove classifier head
        encode.eval()

    if pretraining == "kather":
        feat_path = prep_path.joinpath("patch_features_kather")
        norm_target = data.stain_norm_target()
        normalizer = stainnorm.VahadaneNormalizer()
        normalizer.fit(norm_target)
        predictor = PatchPredictor(pretrained_model="resnet18-kather100k", batch_size=1)

        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: np.array(x.convert("RGB"))),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: einops.repeat(x, "c h w -> b c h w", b=1)),
                transforms.Lambda(lambda x: x.float()),
                # transforms.Lambda(lambda x: x.float().to(device)),
                # transforms.Lambda(lambda x: patch_encoder(x).squeeze()),
            ]
        )

        encode = torch.nn.Sequential(*list(predictor.model.children())[:-1]).to(device)
        encode.eval()

        # encode = encode.to(device)
        patch_tensors = torch.zeros(max_patches, 512)

        # emb = encode(patch)

    num_slides = len(slide_files)
    # extract features
    for slide_count, slide_file in enumerate(coords.keys()):
        save_path = feat_path.joinpath(f"{slide_file}.pt")
        # check if features already extracted
        if os.path.exists(save_path):
            print(f"Features already extracted for slide {slide_file}, skipping...")
            continue

        slide = OpenSlide(site_path.joinpath(f"{slide_file}.svs"))
        print(f"slide {slide_count + 1}/{num_slides}")
        print("file: ", site_path.joinpath(f"{slide_file}.svs"))

        for idx, coord in enumerate(tqdm(coords[slide_file])):
            x, y = coord

            patch_region = slide.read_region((x, y), level=int(level), size=(256, 256))
            patch_region = transform(patch_region)
            patch_region = patch_region.to(device)
            patch_features = encode(patch_region)
            patch_tensors[idx] = patch_features.cpu().detach().squeeze()

        # save features
        if not feat_path.exists():
            feat_path.mkdir(parents=False)
        torch.save(patch_tensors, save_path)


if __name__ == "__main__":
    # data = TCGADataset(config="config/config.yml", dataset="brca")
    #
    # tensors = data[0]
    # for tensor in tensors:
    #     print(tensor.shape)
    config = Config("config/config.yml").read()
    # print(config.to_dict())
    data = TCGASurvivalDataset(**config.to_dict(), dataset="brca", expand=True)
