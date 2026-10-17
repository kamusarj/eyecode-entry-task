#!/usr/bin/env bash
# Creates the `hommi` conda env used for HoMMI data processing + policy training.
# Deviations from the official environment.yml (documented in README):
#  * torch 2.4.1/cu121 instead of 2.1.0 (DINOv3 needs transformers>=4.56, which needs newer torch/hub)
#  * diffusers 0.31 instead of 0.18.2 (0.18 imports huggingface_hub.cached_download, removed in new hub)
#  * pytorch3d replaced by a pure-python shim over the transforms file vendored in the HoMMI repo
set -euo pipefail
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
else
  source "${CONDA_ROOT:-$HOME/anaconda3}/etc/profile.d/conda.sh"
fi
conda env list | grep -q "^hommi " || conda create -y -q -n hommi python=3.10
conda activate hommi
pip install -q torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
# dgl: replaced by hommi_repro/shims/dgl (wheel server returns HTTP 403)
pip install -q numpy==1.26.4 "zarr==2.16.1" "numcodecs==0.11.0" "imagecodecs==2023.9.18" \
  "hydra-core==1.2.0" "accelerate==0.34.2" "diffusers==0.31.0" "transformers==4.56.2" "timm==1.0.15" \
  "einops==0.6.1" "dill==0.3.7" wandb av "opencv-contrib-python==4.7.0.72" scipy numba matplotlib pandas threadpoolctl filelock \
  scikit-image pyyaml transforms3d numpy-quaternion tqdm open3d "huggingface_hub>=0.34" safetensors \
  mujoco==3.3.0 mink "qpsolvers[daqp]" imageio imageio-ffmpeg pydantic \
  lmdb scikit-learn colorama plotly PyExifTool==0.5 easydict==1.9 cloudpickle click pyzmq qrcode "lxml<5" shapely atomics
echo "hommi env ready"
pip install -q --no-deps -e "$(cd "$(dirname "$0")/.." && pwd)/third_party/hommi"   # official env.yml: `-e .`
