"""训练保存与参数统计工具。"""

import torch
import transformers


def print_component_stats(name, trainable, total):
    """打印单个模型组件的参数数量和可训练比例。"""
    ratio = 100 * trainable / total if total > 0 else 0.0
    print(
        f"{name:20} | "
        f"Trainable: {trainable:>12,} | "
        f"Total: {total:>12,} | "
        f"Ratio: {ratio:>6.2f}%"
    )


def numel(p):
    """统计参数量并兼容 DeepSpeed ZeRO 参数。"""
    return p.ds_numel if hasattr(p, "ds_numel") else p.numel()


def print_trainable_parameters(model, training_args):
    """按视觉塔、合并层、语言模型和 LoRA 分组打印可训练参数。"""
    total_params = 0
    trainable_params = 0
    total_params_non_lora = 0
    trainable_params_non_lora = 0

    vision_encoder_total = 0
    vision_encoder_trainable = 0
    merger_total = 0
    merger_trainable = 0
    llm_total = 0
    llm_trainable = 0
    lora_params = 0

    if training_args.lora_enable and hasattr(model, "base_model"):
        model = model.base_model.model

    for name, param in model.named_parameters():
        param_count = numel(param)
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

        if "lora_" in name:
            lora_params += param_count
            assert param.requires_grad, f"LoRA parameter {name} should be trainable"
            continue

        total_params_non_lora += param_count
        if param.requires_grad:
            trainable_params_non_lora += param_count

        if name.startswith(("visual.merger", "model.visual.merger")):
            merger_total += param_count
            if param.requires_grad:
                merger_trainable += param_count
        elif name.startswith(("visual.", "model.visual.")):
            vision_encoder_total += param_count
            if param.requires_grad:
                vision_encoder_trainable += param_count
        elif name.startswith("model.") or name.startswith("lm_head"):
            llm_total += param_count
            if param.requires_grad:
                llm_trainable += param_count
        else:
            print(f"Unrecognized parameter name: {name}.")

    print("=" * 80)
    print("MODEL PARAMETER ANALYSIS")
    print("=" * 80)
    print_component_stats("Vision Encoder", vision_encoder_trainable, vision_encoder_total)
    print_component_stats("Merger", merger_trainable, merger_total)
    print_component_stats("LLM", llm_trainable, llm_total)
    print_component_stats("Total (non-LoRA)", trainable_params_non_lora, total_params_non_lora)
    print_component_stats("LoRA", lora_params, lora_params)
    print_component_stats("Total (include LoRA)", trainable_params, total_params)
    print("=" * 80)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """在非 DeepSpeed 场景下先搬到 CPU 再调用训练器保存模型。"""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
        trainer.model.config.save_pretrained(output_dir)
