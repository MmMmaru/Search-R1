export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='data/nq_search'

WAND_PROJECT='Search-R1'

# export BASE_MODEL='meta-llama/Llama-3.2-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-em
# export BASE_MODEL='meta-llama/Llama-3.2-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.2-3b-it-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-em
# export BASE_MODEL='meta-llama/Llama-3.1-8B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-llama3.1-8b-it-em

# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-7B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-em
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-it-em

export BASE_MODEL='/root/autodl-tmp/models/Qwen/Qwen3-0.6B'
export EXPERIMENT_NAME=nq-search-r1-grpo-qwen3-0.6b
export RAY_DEBUG_POST_MORTEM=1

# set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])


PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    +data.max_start_length=1000 \
    +data.max_obs_length=1000 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    +algorithm.no_think_rl=false \
    trainer.val_before_train=false\
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_training_steps=200 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    +do_search=true \
    +max_turns=2 \
    +retriever.url="http://127.0.0.1:8000/retrieve" \
    +retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
    