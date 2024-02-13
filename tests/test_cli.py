import pytest
import subprocess


@pytest.mark.slow
def test_download():
    # subprocess.run(f"conda activate castle", shell=True)
    mamba = f"/home/kh701/miniforge3/bin/mamba"
    command = f"{mamba} run -n castle invoke download --dataset tcga --samples 1"
    # command = f"{mamba} run -n castle invoke download --dataset chestx"

    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    assert result.returncode == 0, f"Subprocess failed with message: {result.stderr}"
