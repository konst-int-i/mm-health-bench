from mmhb.loader import MMDataset
import torch
from typing import *
from pathlib import Path
import pandas as pd
import os


class TCGADataset(MMDataset):
    """
    Loads omic and WSI data of specified TCGA site
    """

    def __init__(self, dataset: str):
        super().__init__()
        self.dataset = dataset

        assert "tcga.yml" in os.listdir("config/"), "Missing config file!"

        # self.config =
        pass

    def load_omic(
        self,
        eps: float = 1e-6,
        filter_overlap: bool = True,
    ) -> pd.DataFrame:
        """
        Loads in omic data and returns a dataframe and filters depending on which whole slide images
        are available, such that only samples with both omic and WSI data are kept.
        Also calculates the discretised survival time for each sample.
        Args:
            eps (float): Epsilon value to add to min and max survival time to ensure all samples are included

        Returns:
            pd.DataFrame: Dataframe with omic data and discretised survival time (target)
        """
        data_path = Path(self.config.tcga_path).joinpath(
            f"omic/tcga_{self.dataset}_all_clean.csv.zip"
        )
        df = pd.read_csv(
            data_path, compression="zip", header=0, index_col=0, low_memory=False
        )

        # handle missing values
        num_nans = df.isna().sum().sum()
        nan_counts = df.isna().sum()[df.isna().sum() > 0]
        df = df.fillna(df.mean(numeric_only=True))
        print(f"Filled {num_nans} missing values with mean")
        print(f"Missing values per feature: \n {nan_counts}")

        # filter samples for which there are no slides available
        if self.filter_overlap:
            slides_available = self.slide_ids
            omic_available = [id[:-4] for id in df["slide_id"]]
            overlap = set(slides_available) & set(omic_available)
            print(f"Slides available: {len(slides_available)}")
            print(f"Omic available: {len(omic_available)}")
            print(f"Overlap: {len(overlap)}")
            if len(slides_available) < len(omic_available):
                print(
                    f"Filtering out {len(omic_available) - len(slides_available)} samples for which there are no omic data available"
                )
                overlap_filter = [id + ".svs" for id in overlap]
                df = df[df["slide_id"].isin(overlap_filter)]
            elif len(slides_available) > len(omic_available):
                print(
                    f"Filtering out {len(slides_available) - len(omic_available)} samples for which there are no slides available"
                )
                self.slide_ids = overlap
            else:
                print("100% modality overlap, no samples filtered out")

        # assign target column (high vs. low risk in equal parts of survival)
        label_col = "survival_months"
        if self.subset == "all":
            df["y_disc"] = pd.qcut(df[label_col], q=self.n_bins, labels=False).values
        else:
            if self.subset == "censored":
                subset_df = df[df["censorship"] == 1]
            elif self.subset == "uncensored":
                subset_df = df[df["censorship"] == 0]
            # take q_bins from uncensored patients
            disc_labels, q_bins = pd.qcut(
                subset_df[label_col], q=self.n_bins, retbins=True, labels=False
            )
            q_bins[-1] = df[label_col].max() + eps
            q_bins[0] = df[label_col].min() - eps
            # use bin cuts to discretize all patients
            df["y_disc"] = pd.cut(
                df[label_col],
                bins=q_bins,
                retbins=False,
                labels=False,
                right=False,
                include_lowest=True,
            ).values

        df["y_disc"] = df["y_disc"].astype(int)

        if self.log_dir is not None:
            df.to_csv(
                self.log_dir.joinpath(f"{self.dataset}_omic_overlap.csv.zip"),
                compression="zip",
            )

        return df
