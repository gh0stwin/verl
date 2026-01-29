#!/bin/bash

ARGS_LONG="cpu:,mem:,gpu:"
if ! OPTIONS=$(getopt \
  -o "" \
  -l $ARGS_LONG \
  -- "$@"); then
  exit 1
fi

eval set -- "$OPTIONS"

while [ $# -gt 0 ]; do
  case $1 in
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
#SBATCH --gres=gpu:RTX_PRO_6000_96GB:$USER_GPU               # number of gpus per node
#SBATCH --export=ALL                    # export all environment variables
#SBATCH --time=48:00:00                 # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin               # send mail when job begins
#SBATCH --mail-type=end                 # send mail when job ends
#SBATCH --mail-type=fail                # send mail if job fails
#SBATCH --mail-user=fabiovital@tecnico.ulisboa.pt
#SBATCH --account=inesc
#SBATCH --partition=gaips_phd
#SBATCH --nodelist=a01
#IGNORE#SBATCH --dependency=after:16261

# set -x  # Echo commands being executed (useful for debugging)

TMP_DIR="/scratch/datasets/u021032/home"
PROJ_DIR="\${TMP_DIR}/rlvr"

# Unset conda vars
unset CONDA_ENVS_PATH
unset CONDA_EXE
unset CONDA_PYTHON_EXE
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

# Move $HOME to the job's folder created by slurm
cd \$TMP_DIR
source .bashrc

# Copy git repo (already in cfs) and install conda env from $(environment.yml)
cd \$HOME/verl
git pull

# Activate env
export USER_CONDA_ENV=verl  # conda env name
conda activate \$USER_CONDA_ENV
source .venv/bin/activate

unset ROCR_VISIBLE_DEVICES
./recipe/corrected_adv/run_baseline.sh

EOF
