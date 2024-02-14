import pandas as pd
import einops
from mmhb.loader import MMDataset
from torchvision import transforms
import numpy as np
from typing import Union, List
from pathlib import Path
import os
from os import listdir
from os.path import isfile, join
from xml.dom import minidom
from PIL import Image
import xml.etree.ElementTree as ET
from transformers import BertTokenizer


class ChestXDataset(MMDataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        expand: bool = False,
        modalities: List = ["images", "reports"],
        max_seq_length: int = 512,
        **kwargs
    ):
        super().__init__(data_path, expand, modalities, **kwargs)

        self.max_seq_length = max_seq_length
        self.dataset = "chestx"
        self.df = self.load_text_labels(
            self.data_path.joinpath("proc/full_dataset.csv")
        )

        self.reports = self.df["report"]
        self.img_name = self.df["id"]
        self.targets = self.df["list"]

        # image preprocessing utils
        self.preprocess_img = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        # report preprocessing utils
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=False
        )

        # assign single sample
        self.tensor, self.target = self.__getitem__(0)

    def __getitem__(self, idx):
        name = self.img_name[idx].split(".")[0]

        # load image
        img_path = self.data_path.joinpath("proc/images", self.img_name[idx])
        image = Image.open(img_path)
        image = self.preprocess_img(image)
        # reshape to expected dims
        image = einops.rearrange(image, "c h w -> h w c")

        # load report
        report = self.vectorize_report(self.reports[idx])
        if not self.expand:
            report = report.squeeze()

        tensors = [image, report]

        return tensors, self.targets[idx]

    def __len__(self):
        return len(self.targets)

    def load_text_labels(self, path):
        txt_gt = pd.read_csv(path)
        txt_gt["list"] = txt_gt[txt_gt.columns[2:]].values.tolist()
        txt_gt = txt_gt[["id", "report", "list"]].copy()
        return txt_gt

    def vectorize_report(self, report):
        return self.tokenizer.encode(
            report,
            max_length=self.max_seq_length,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )


def _create_report(img_names_list_, report_list_, gt_list_, save_add):
    pd.DataFrame(
        {
            "id": img_names_list_,
            "report": report_list_,
            "Atelectasis": gt_list_[:, 0],
            "Cardiomegaly": gt_list_[:, 1],
            "Consolidation": gt_list_[:, 2],
            "Edema": gt_list_[:, 3],
            "Enlarged-Cardiomediastinum": gt_list_[:, 4],
            "Fracture": gt_list_[:, 5],
            "Lung-Lesion": gt_list_[:, 6],
            "Lung-Opacity": gt_list_[:, 7],
            "No-Finding": gt_list_[:, 8],
            "Pleural-Effusion": gt_list_[:, 9],
            "Pleural_Other": gt_list_[:, 10],
            "Pneumonia": gt_list_[:, 11],
            "Pneumothorax": gt_list_[:, 12],
            "Support-Devices": gt_list_[:, 13],
        }
    ).to_csv(save_add, index=False)


def preprocess_chestx(raw_path, proc_path):
    """
    Note: Credit to MONAI for their implementaiton of this preprocessing pipeline
    """

    report_file_add = raw_path.joinpath("NLMCXR_reports/ecgen-radiology")
    img_file_add = raw_path.joinpath("NLMCXR_png")
    npy_add = raw_path.joinpath("TransChex_openi")

    img_save_add = proc_path.joinpath("images")
    report_train_save_add = proc_path.joinpath("train.csv")
    report_val_save_add = proc_path.joinpath("validation.csv")
    report_test_save_add = proc_path.joinpath("test.csv")
    report_all_save_add = proc_path.joinpath("full_dataset.csv")

    if not os.path.isdir(img_save_add):
        os.makedirs(img_save_add)
    report_files = [
        f for f in listdir(report_file_add) if isfile(join(report_file_add, f))
    ]

    train_data = np.load(npy_add.joinpath("train.npy"), allow_pickle=True).item()
    train_data_id = train_data["id_GT"]
    train_data_gt = train_data["label_GT"]

    val_data = np.load(npy_add.joinpath("validation.npy"), allow_pickle=True).item()
    val_data_id = val_data["id_GT"]
    val_data_gt = val_data["label_GT"]

    test_data = np.load(npy_add.joinpath("test.npy"), allow_pickle=True).item()
    test_data_id = test_data["id_GT"]
    test_data_gt = test_data["label_GT"]

    all_cases = np.union1d(np.union1d(train_data_id, val_data_id), test_data_id)

    img_names_list_train = []
    img_names_list_val = []
    img_names_list_test = []

    report_list_train = []
    report_list_val = []
    report_list_test = []

    gt_list_train = []
    gt_list_val = []
    gt_list_test = []

    for file in report_files:
        print("Processing {}".format(file))
        add_xml = os.path.join(report_file_add, file)
        docs = minidom.parse(add_xml)
        tree = ET.parse(add_xml)
        for node in tree.iter("AbstractText"):
            i = 0
            for elem in node.iter():
                if elem.attrib["Label"] == "FINDINGS":
                    if elem.text == None:
                        report = "FINDINGS : "
                    else:
                        report = "FINDINGS : " + elem.text
                elif elem.attrib["Label"] == "IMPRESSION":
                    if elem.text == None:
                        report = report + " IMPRESSION : "
                    else:
                        report = report + " IMPRESSION : " + elem.text
        images = docs.getElementsByTagName("parentImage")
        for i in images:
            img_name = i.getAttribute("id") + ".png"
            if img_name in all_cases:
                Image.open(os.path.join(img_file_add, img_name)).resize(
                    (512, 512)
                ).save(os.path.join(img_save_add, img_name))
                if img_name in train_data_id:
                    img_names_list_train.append(img_name)
                    report_list_train.append(report)
                    gt_list_train.append(
                        train_data_gt[np.where(train_data_id == img_name)[0][0]]
                    )
                elif img_name in val_data_id:
                    img_names_list_val.append(img_name)
                    report_list_val.append(report)
                    gt_list_val.append(
                        val_data_gt[np.where(val_data_id == img_name)[0][0]]
                    )
                elif img_name in test_data_id:
                    img_names_list_test.append(img_name)
                    report_list_test.append(report)
                    gt_list_test.append(
                        test_data_gt[np.where(test_data_id == img_name)[0][0]]
                    )

    img_names_list_all = img_names_list_train + img_names_list_val + img_names_list_test
    report_list_all = report_list_train + report_list_val + report_list_test
    gt_list_all = np.concatenate([gt_list_train, gt_list_val, gt_list_test], axis=0)

    datasets = [
        {
            "save_add": report_all_save_add,
            "img_name": np.array(img_names_list_all),
            "report": np.array(report_list_all),
            "gt": np.array(gt_list_all),
        },
        {
            "save_add": report_train_save_add,
            "img_name": np.array(img_names_list_train),
            "report": np.array(report_list_train),
            "gt": np.array(gt_list_train),
        },
        {
            "save_add": report_val_save_add,
            "img_name": np.array(img_names_list_val),
            "report": np.array(report_list_val),
            "gt": np.array(gt_list_val),
        },
        {
            "save_add": report_test_save_add,
            "img_name": np.array(img_names_list_test),
            "report": np.array(report_list_test),
            "gt": np.array(gt_list_test),
        },
    ]
    for dataset in datasets:
        _create_report(
            dataset["img_name"], dataset["report"], dataset["gt"], dataset["save_add"]
        )

    print("Processed Dataset Files Are Saved !")


if __name__ == "__main__":
    chestx = ChestXDataset(
        data_path=Path("data/chestx"),
    )
    [img, report], label = chestx[0]
    print(img.shape)
    print(report)
    print(report.shape)
    print(label)
    # unnormalized_img = sample_img.permute(1, 2, 0).numpy() * 0.5 + 0.5
    # plt.imshow(unnormalized_img)
    # plt.axis('off')
    # plt.show()

    # data_path = Path("data/chestx/")
    #
    # raw_path = Path("./data/chestx/raw/")
    # proc_path = Path(".data/chestx/proc/")
    #
    # preprocess_chestx(raw_path, proc_path)
