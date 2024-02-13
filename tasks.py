from invoke import task
from typing import *
import os
from pathlib import Path
import pandas as pd
from mmhb.utils import detect_os


@task
def download(
    c,
    dataset: str,
    sites: Optional[List[str]] = ["brca"],
    data_dir: Optional[str] = None,
):
    if data_dir is None:
        # set default
        data_dir = Path("data/{dataset}")

    valid_datasets = ["tcga", "mimic", "chest-x"]
    assert (
        dataset in valid_datasets
    ), f"Invalid dataset, specify one of {valid_datasets}"

    if dataset == "tcga":
        install_tcga_deps(c)
        download_tcga(c, sites, data_dir)


@task
def download_tcga(c, sites: str, data_dir: Path, samples: int = None):
    valid_sites = ["brca", "blca", "kirp", "ucec", "hnsc", "paad", "luad", "lusc"]
    # check that all sites are in valid_sites
    assert all(
        site in valid_sites for site in sites
    ), f"Invalid site, must be one of {valid_sites}"

    for site in sites:
        download_dir = data_dir.joinpath(f"tcga/wsi/{site}")
        if not download_dir.exists():
            download_dir.mkdir(parents=True)

        manifest_path = Path(
            f"./data/tcga/gdc_manifests/filtered/{site}_wsi_manifest_filtered.txt"
        )

        if samples is not None:
            manifest = pd.read_csv(manifest_path, sep="\t")
            manifest = manifest.sample(n=int(samples), random_state=42)
            tmp_path = manifest_path.parent.joinpath(f"{site}_tmp.txt")
            manifest.to_csv(tmp_path, sep="\t", index=False)
            print(f"Downloading {manifest.shape[0]} files from {site} dataset...")
            c.run(f"gdc-client download -m {tmp_path} -d {download_dir}")
            # cleanup
            os.remove(tmp_path)
        else:
            command = f"gdc-client download -m {manifest_path} -d {download_dir}"
            try:
                c.run(command)
            except Exception as e:
                print(f"Error occurred: {e}")
                print(f"Command: {command}")

        flatten_tcga_dir(c, download_dir)  # flatten directory structure after download
    return None


@task
def install_tcga_deps(c):
    # check if gdc client is installed
    try:
        c.run("gdc-client --version")
        print(f"gdc-client already installed at {os.getcwd()}")
    except:
        pass

    system = detect_os()

    print(f"Installing gdc-client for {system}...")
    if system == "linux":
        c.run(
            "curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip "
            "--output gdc-client.zip"
        )
        c.run("unzip gdc-client.zip")
    if system == "mac":
        c.run(
            "curl -0 https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_OSX_x64.zip "
            "--output gdc-client.zip"
        )
        c.run("unzip gdc-client.zip")
    elif system == "windows":
        raise NotImplementedError

    print(f"Installed gdc-client at {os.getcwd()}")
    # cleanup
    os.remove("gdc-client.zip")


@task
def flatten_tcga_dir(c, download_dir: Path):
    """
    Flattens directory structure for WSI images after download using the GDC client from
     `data_dir/*.svs` instead of `data_dir/hash_subdir/*.svs`.
    Args:
        c:
        dataset:
        config:

    Returns:
    """
    # flatten directory structure
    c.run(f"find {download_dir} -type f -name '*.svs' -exec mv {{}} {download_dir} \;")
    # remove everything that's not a .svs file
    c.run(f"find {download_dir} ! -name '*.svs' -delete")
