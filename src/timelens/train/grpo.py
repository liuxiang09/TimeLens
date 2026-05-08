"""视频时序定位 GRPO 训练入口。"""

import os
import pathlib

import torch
from transformers import HfArgumentParser

from src.timelens.rewards import load_reward_funcs
from src.timelens.data.training_data import HybridDataset
from src.training.args import DataArguments, GRPOArguments, ModelArguments
from src.training.grpo_trainer import QwenvlGRPOTrainer
from src.training.lora import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
)
from src.training.model_setup import (
    apply_lora_if_needed,
    compute_dtype_from_args,
    configure_gradient_checkpointing,
    configure_llm,
    configure_vision_tower,
    load_model_processor_and_adapter,
    normalize_kbit_module_dtypes,
    parse_lora_excludes,
    prepare_kbit_training,
    validate_lora_freeze_args,
)
from src.training.save import (
    print_trainable_parameters,
    safe_save_model_for_hf_trainer,
)


def _configure_grpo_processor(processor):
    """补齐 GRPO 训练器需要的处理器和分词器属性。"""
    processor.pad_token_id = processor.tokenizer.pad_token_id
    processor.bos_token_id = processor.tokenizer.bos_token_id
    processor.eos_token_id = processor.tokenizer.eos_token_id
    processor.pad_token = processor.tokenizer.pad_token


def train():
    """解析命令行参数并启动视频时序定位 GRPO。"""
    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    validate_lora_freeze_args(training_args)
    parse_lora_excludes(training_args)

    compute_dtype = compute_dtype_from_args(training_args)
    model, processor, _adapter, _ = load_model_processor_and_adapter(
        model_args,
        training_args,
        processor_kwargs={"padding_side": "left"},
        bnb_skip_lm_head=False,
    )
    _configure_grpo_processor(processor)
    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    model = prepare_kbit_training(model, training_args)
    configure_gradient_checkpointing(model, training_args)
    model, peft_config = apply_lora_if_needed(model, training_args, merge_on_create=False)

    if training_args.local_rank in (0, -1):
        print_trainable_parameters(model, training_args=training_args)

    normalize_kbit_module_dtypes(model, training_args)

    trainer = QwenvlGRPOTrainer(
        model=model,
        processing_class=processor,
        train_dataset=HybridDataset(
            processor,
            model.config,
            model_args,
            data_args,
            training_args,
            training_mode="grpo",
        ),
        reward_funcs=load_reward_funcs(training_args.reward_funcs),
        args=training_args,
        peft_config=peft_config,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=False
        )
        if training_args.local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

    if training_args.local_rank in (0, -1):
        import shutil

        checkpoints_sorted = trainer._sorted_checkpoints(
            use_mtime=False, output_dir=training_args.output_dir
        )
        for checkpoint in checkpoints_sorted[-1:]:
            print(f"Deleting older checkpoint [{checkpoint}].")
            shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    train()
