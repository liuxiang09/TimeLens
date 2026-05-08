"""视频时序定位 SFT 训练入口。"""

import os
import pathlib
import random

import torch
from transformers import HfArgumentParser

from src.data.collator import HybridDataCollator
from src.timelens.data.training_data import HybridDataset
from src.training.args import DataArguments, ModelArguments, TrainingArguments
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
from src.training.sft_trainer import QwenSFTTrainer


def train():
    """解析命令行参数并启动视频时序定位 SFT。"""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    validate_lora_freeze_args(training_args)
    parse_lora_excludes(training_args)

    compute_dtype = compute_dtype_from_args(training_args)
    model, processor, _adapter, _ = load_model_processor_and_adapter(
        model_args,
        training_args,
    )
    model.config.use_cache = False
    configure_llm(model, training_args)
    configure_vision_tower(model, training_args, compute_dtype, training_args.device)

    model = prepare_kbit_training(model, training_args)
    configure_gradient_checkpointing(model, training_args)
    model, _peft_config = apply_lora_if_needed(model, training_args, merge_on_create=True)

    if training_args.local_rank in (0, -1):
        print_trainable_parameters(model, training_args=training_args)

    normalize_kbit_module_dtypes(model, training_args)
    random.seed(training_args.seed)

    trainer = QwenSFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        data_collator=HybridDataCollator(processor.tokenizer),
        train_dataset=HybridDataset(
            processor,
            model.config,
            model_args,
            data_args,
            training_args,
        ),
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
            model.named_parameters(), require_grad_only=True
        )
        if training_args.local_rank in (0, -1):
            output_dir_lora = os.path.join(training_args.output_dir, "lora")
            model.config.save_pretrained(output_dir_lora)
            model.save_pretrained(output_dir_lora, state_dict=state_dict)
            processor.save_pretrained(output_dir_lora)
            torch.save(
                non_lora_state_dict,
                os.path.join(output_dir_lora, "non_lora_state_dict.bin"),
            )
            print("Merging LoRA weights into base model ...")
            merged_model = model.merge_and_unload()
            output_dir_merged = os.path.join(training_args.output_dir, "merged")
            merged_model.save_pretrained(output_dir_merged, safe_serialization=True)
            processor.save_pretrained(output_dir_merged)
            print(f"Merged model saved to {output_dir_merged}")
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)

    if training_args.local_rank in (0, -1):
        import shutil

        checkpoints_sorted = trainer._sorted_checkpoints(
            use_mtime=False, output_dir=training_args.output_dir
        )
        for checkpoint in checkpoints_sorted:
            shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    train()
