```bash
conda env create -f environment.yml -y
conda activate rlvr
uv venv .venv --python 3.13 --seed
source .venv/bin/activate
# TODO: environment.yml -> need to install 12.8 :(
uv pip install vllm==0.14.1 --torch-backend=cu130
# uv pip install deepspeed-kernels
# DS_BUILD_OPS=1 uv pip install deepspeed==0.16.8
# uv pip install oat-llm==0.2.0
uv pip install -e .
uv sync
pip install nvidia-nccl-cu12==2.27.7
```
