#!/bin/bash

ARGS_LONG="script:,cpu:,mem:,gpu:"
if ! OPTIONS=$(getopt \
  -o "" \
  -l $ARGS_LONG \
  -- "$@"); then
  exit 1
fi

eval set -- "$OPTIONS"

while [ $# -gt 0 ]; do
  case $1 in
  --script)
    USER_SCRIPT=${2}
    shift
    ;;
  --cpu)
    USER_CPU=${2}
    shift
    ;;
  --mem)
    USER_MEM=${2}
    shift
    ;;
  --gpu)
    USER_GPU=${2}
    shift
    ;;
  --)
    shift
    break
    ;;
  -*)
    echo "$0: error - unrecognized option $1" 1>&2
    exit 1
    ;;
  *) break ;;
  esac
  shift
done

if [ -z "$USER_CPU" ]; then
  USER_CPU=8
fi

if [ -z "$USER_MEM" ]; then
  USER_MEM=64
fi

if [ -z "$USER_GPU" ]; then
  USER_GPU="2"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name="rlvr" # create a short name for your job
#SBATCH --nodes=1                       # node count
#SBATCH --ntasks=1                      # total number of tasks across all nodes
#SBATCH --cpus-per-task=$USER_CPU            # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=${USER_MEM}G                   # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:$USER_GPU               # number of gpus per node
#SBATCH --export=ALL                    # export all environment variables
#SBATCH --time=5-00:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send mail when job begins
#SBATCH --mail-type=end                 # send mail when job ends
#SBATCH --mail-type=fail                # send mail if job fails
#SBATCH --mail-user=fabiovital@tecnico.ulisboa.pt
#SBATCH --account=inesc
#SBATCH --partition=gaips_gpu_high
#IGNORE_SBATCH --nodelist=a01
#IGNORE_SBATCH --dependency=after:16261

# set -x  # Echo commands being executed (useful for debugging)

TMP_DIR="/tmp/verl_u021032"
HOME="\${TMP_DIR}"

# Unset conda vars
unset CONDA_ENVS_PATH
unset CONDA_EXE
unset CONDA_PYTHON_EXE
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

if [ ! -d "\$TMP_DIR" ]: then
  mkdir \$TMP_DIR
  chmod 700 -R \$TMP_DIR
  cd \$TMP_DIR
  touch .bashrc
  echo "unset CONDA_ENVS_PATH" >>\$HOME/.bashrc
  echo "unset CONDA_EXE" >>\$HOME/.bashrc
  echo "unset CONDA_PYTHON_EXE" >>\$HOME/.bashrc
  echo "unset CONDA_PREFIX" >>\$HOME/.bashrc
  echo "unset CONDA_DEFAULT_ENV" >>\$HOME/.bashrc
  echo "export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin:/usr/local/games:/usr/games" >>\$HOME/.bashrc
  echo "export HOME=\$TMP_DIR" >>\$HOME/.bashrc
  source .bashrc
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-py313_25.3.1-1-Linux-x86_64.sh
  bash \$HOME/Miniconda3-py313_25.3.1-1-Linux-x86_64.sh -b -p \$HOME/miniconda3
  miniconda3/bin/conda init bash
  sed -i "8i\export CONDA_PREFIX=\${TMP_DIR}/miniconda3" \$HOME/.bashrc
  source .bashrc
  cp -r /cfs/home/u021032/dev/storage .
  rsync -av --progress --exclude={'.git','slurm-*.out','.venv','docs','notebooks','examples','docker','tests','.gemini','.github'} /cfs/home/u021032/dev/verl \$HOME
  conda update -n base -c defaults conda -y
  cd verl
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
  export USER_CONDA_ENV=verl
  conda activate \$USER_CONDA_ENV
  source .venv/bin/activate
fi

# Move $HOME to the job's folder created by slurm
cd \$TMP_DIR
source .bashrc
rsync -av --progress --exclude={'.git','slurm-*.out','.venv','docs','notebooks','examples','docker','tests','.gemini','.github'} /cfs/home/u021032/dev/verl \$HOME
cd ./verl

# Activate env
export USER_CONDA_ENV=verl  # conda env name
conda activate \$USER_CONDA_ENV
source .venv/bin/activate

unset ROCR_VISIBLE_DEVICES
./recipe/cat_critic/${USER_SCRIPT}.sh

EOF
