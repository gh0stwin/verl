#!/bin/bash

conda env create -n verl python=3.12 -y
conda activate verl
conda install conda-forge::uv==0.9.26 -y
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm==0.12.0 --torch-backend=cpu
uv pip install ruff==0.14.14
uv pip install mypy==1.19.1
uv pip install pyright==1.1.408
uv pip install "tensordict>=0.8.0,<=0.10.0,!=0.9.0" datasets hydra-core notebook
