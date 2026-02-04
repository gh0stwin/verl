#!/bin/bash
set -xeuo pipefail

SEED=$((((RANDOM << 30) | (RANDOM << 15) | RANDOM) & 0x7fffffff))
project_name='verl'
exp_name="CC-DeepSeek-R1-1.5B-${SEED}"

adv_estimator=gae_cat

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=64
gen_prompt_bsz=$((train_prompt_bsz * 2))
n_resp_per_prompt=4
train_prompt_mini_bsz=64

# Ray
PORT=$(($RANDOM % 2000 + 12000))
RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:${PORT}"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths
STORAGE_HOME=${STORAGE_HOME:-"${HOME}/storage"}
MODEL_PATH=${MODEL_PATH:-"${STORAGE_HOME}/models/DeepSeek-R1-Distill-Qwen-1.5B"}
CKPTS_DIR=${CKPTS_DIR:-"${WORKING_DIR}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${STORAGE_HOME}/data/deepscaler/deepscaler-train-subset.parquet"}
TEST_FILE=${TEST_FILE:-"${STORAGE_HOME}/data/deepscaler/aime-2024.parquet"}

# Algorithm
GPU_UTIL=${GPU_UTIL:-"0.6"}
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=1.0

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
param_offload=True
optim_offload=True
gen_tp=1

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export TP_SOCKET_IFNAME=enp1s0f0np0

HYDRA_FULL_ERROR=1 python3 -m recipe.cat_critic.main_dapo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.truncation='left' \
  data.max_prompt_length=${max_prompt_length} \
  data.max_response_length=${max_response_length} \
  data.gen_batch_size=${gen_prompt_bsz} \
  data.train_batch_size=${train_prompt_bsz} \
  data.shuffle=True \
  data.filter_overlong_prompts=True \
  data.seed=$((SEED + 1)) \
  actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
  algorithm.adv_estimator=${adv_estimator} \
  algorithm.use_kl_in_reward=${use_kl_in_reward} \
  algorithm.filter_groups.enable=${enable_filter_groups} \
  algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
  algorithm.filter_groups.metric=${filter_groups_metric} \
  algorithm.kl_ctrl.kl_coef=${kl_coef} \
  actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
  actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
  actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
  actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
  actor_rollout_ref.actor.fsdp_config.param_offload=${param_offload} \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optim_offload} \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
  actor_rollout_ref.rollout.temperature=${temperature} \
  actor_rollout_ref.rollout.top_p=${top_p} \
  actor_rollout_ref.rollout.top_k="${top_k}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
  actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
  actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=4 \
  actor_rollout_ref.rollout.name=vllm \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max_model_len=$((max_prompt_length + max_response_length)) \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.seed=${SEED} \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL} \
  actor_rollout_ref.ref.fsdp_config.param_offload=${param_offload} \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
  reward_model.reward_manager=dler \
  reward_model.overlong_buffer.enable=True \
  reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
  reward_model.overlong_buffer.log=True \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${exp_name}" \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes="${NNODES}" \
  trainer.val_before_train=True \
  trainer.test_freq=5 \
  trainer.save_freq=10 \
  trainer.total_epochs=10 \
  trainer.total_training_steps=200 \
  trainer.default_local_dir="${CKPTS_DIR}" \
  trainer.resume_mode=auto
