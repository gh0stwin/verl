#!/bin/bash

conda env create -f environment.yml -y
conda activate verl
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install --no-cache-dir vllm==0.12.0 --torch-backend=cu128
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata ray[default] codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler pytest py-spy pre-commit ruff tensorboard
uv pip install git+https://github.com/ShaohonChen/PyExt.git@py311support
uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl && uv pip install --no-cache-dir flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
uv pip install --no-cache-dir flashinfer-python==0.3.1
uv pip install opencv-python
uv pip install opencv-fixer && python -c "from opencv_fixer import AutoFix; AutoFix()"
uv pip install nvidia-cudnn-cu12==9.10.2.21
uv pip install --no-build-isolation flash-attn
uv pip install "packaging>=20.0" uvicorn fastapi latex2sympy2_extended math_verify
uv pip install "numpy<2.0.0"
