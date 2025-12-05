
from verl.model_merger.base_model_merger import ModelMergerConfig
from verl.model_merger.fsdp_model_merger import FSDPModelMerger

if __name__ == "__main__":
    config = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        local_dir="verl_checkpoints/nq-search-r1-ppo-qwen3-0.6b/global_step_100/actor",
        target_dir="./trained_100steps/huggingface",
        hf_model_config_path="verl_checkpoints/nq-search-r1-ppo-qwen3-0.6b/global_step_100/actor/huggingface",
    )
    merger = FSDPModelMerger(config)
    merger.merge_and_save()