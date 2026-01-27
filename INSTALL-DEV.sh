#!/bin/bash

conda env create -n rlvr python=3.13 -y
conda activate rlvr
conda install conda-forge::uv==0.9.26 -y
uv venv .venv --python 3.13 --seed
source .venv/bin/activate
uv pip install ruff==0.14.14
uv pip install mypy==1.19.1
uv pip install pyright==1.1.408
