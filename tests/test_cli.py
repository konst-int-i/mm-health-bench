import pytest
import subprocess


def test_download():
    # subprocess.run(f"conda activate castle", shell=True)
    mamba = f"/home/kh701/miniforge3/bin/mamba"
    result = subprocess.run(
        f"{mamba} run -n castle invoke download --dataset tcga --samples 1",
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    assert result.returncode == 0, f"Subprocess failed with message: {result.stderr}"
