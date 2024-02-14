import pandas as pd
import einops
from mmhb.loader import MMDataset
from mmhb.utils import setup_logging, RepeatTransform, RearrangeTransform, Config
import torch
import os
from typing import Union, List, Tuple
from pathlib import Path

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
            tensor = self.load_patches(slide_id=self.slide_ids[idx])
            if self.expand:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)

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


if __name__ == "__main__":
    # data = TCGADataset(config="config/config.yml", dataset="brca")
    #
    # tensors = data[0]
    # for tensor in tensors:
    #     print(tensor.shape)
    config = Config("config/config.yml").read()
    # print(config.to_dict())
    data = TCGASurvivalDataset(**config.to_dict(), dataset="brca", expand=True)
