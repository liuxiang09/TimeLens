"""训练入口共用的模型加载、冻结和量化配置。"""

import ast

import torch
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

from src.models.loader import resolve_processor_source
from src.models.registry import get_adapter
from src.training.lora import find_target_linear_names


def parse_lora_excludes(training_args):
    """解析 LoRA 排除模块字符串并补齐视觉塔默认排除规则。"""
    if training_args.lora_namespan_exclude is not None:
        training_args.lora_namespan_exclude = ast.literal_eval(
            training_args.lora_namespan_exclude
        )
    else:
        training_args.lora_namespan_exclude = []

    if not training_args.vision_lora:
        training_args.lora_namespan_exclude += ["visual"]


def validate_lora_freeze_args(training_args):
    """校验 LoRA 与冻结参数组合是否合法。"""
    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")
    if not training_args.lora_enable and training_args.vision_lora:
        raise ValueError(
            "training_args.vision_lora is enabled but lora_enable is disabled."
        )
    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")


def compute_dtype_from_args(training_args):
    """根据 fp16/bf16 参数计算模型加载数据类型。"""
    if training_args.fp16:
        return torch.float16
    if training_args.bf16:
        return torch.bfloat16
    return torch.float32


def build_bnb_args(training_args, compute_dtype, skip_lm_head=True):
    """构造 bitsandbytes 4/8 位加载参数。"""
    if training_args.bits not in [4, 8]:
        return {}
    skip_modules = ["visual", "lm_head"] if skip_lm_head else ["visual"]
    return {
        "device_map": {"": training_args.device},
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_skip_modules=skip_modules,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,
        ),
    }


def set_requires_grad(parameters, requires_grad):
    """批量设置参数是否参与训练。"""
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    """配置视觉塔数据类型、设备与冻结状态。"""
    vision_tower = getattr(model, "visual", None)
    if vision_tower is None:
        return
    vision_tower.to(dtype=compute_dtype, device=device)
    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)

    merger = getattr(vision_tower, "merger", None)
    if merger is not None:
        set_requires_grad(merger.parameters(), not training_args.freeze_merger)


def configure_llm(model, training_args):
    """配置语言模型主体和输出头的冻结状态。"""
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not training_args.freeze_llm)


def prepare_kbit_training(model, training_args):
    """在 4/8 位训练时启用 PEFT 的低比特训练准备。"""
    if training_args.bits not in [4, 8]:
        return model
    model.config.torch_dtype = (
        torch.float32
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    from peft import prepare_model_for_kbit_training

    return prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=training_args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )


def configure_gradient_checkpointing(model, training_args):
    """启用梯度检查点所需的输入梯度。"""
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}


def build_lora_config(model, training_args):
    """根据训练参数构造 LoRA 配置。"""
    return LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_target_linear_names(
            model,
            lora_namespan_exclude=training_args.lora_namespan_exclude,
            num_lora_modules=training_args.num_lora_modules,
            local_rank=training_args.local_rank,
        ),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
    )


def apply_lora_if_needed(model, training_args, merge_on_create=True):
    """按需向模型注入 LoRA，并返回模型与 PEFT 配置。"""
    if not training_args.lora_enable:
        return model, None

    peft_config = build_lora_config(model, training_args)
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    if merge_on_create:
        model = get_peft_model(model, peft_config)
        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True
        if not training_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True
    return model, peft_config


def normalize_kbit_module_dtypes(model, training_args):
    """修正低比特训练中 LoRA、归一化层和嵌入/输出头的数据类型。"""
    if training_args.bits not in [4, 8]:
        return
    from peft.tuners.lora import LoraLayer

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and training_args.bf16:
            module.to(torch.bfloat16)
        if "norm" in name:
            module.to(torch.float32)
        if "lm_head" in name or "embed_token" in name:
            if hasattr(module, "weight") and training_args.bf16 and module.weight.dtype == torch.float32:
                module.to(torch.bfloat16)


def load_model_processor_and_adapter(
    model_args, training_args, processor_kwargs=None, bnb_skip_lm_head=True
):
    """加载模型、处理器和 Qwen 适配器。"""
    processor_kwargs = processor_kwargs or {}
    processor_source = resolve_processor_source(
        model_args.model_name_or_path,
        model_args.processor_path,
    )
    model_args.processor_path = processor_source
    adapter = get_adapter(processor_source, model_args.model_name_or_path, model_args.model_id)
    compute_dtype = compute_dtype_from_args(training_args)
    config = adapter.load_config(model_args.model_name_or_path)
    model = adapter.load_model(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=compute_dtype,
        attn_implementation=(
            "flash_attention_2" if not training_args.disable_flash_attn2 else "sdpa"
        ),
        trust_remote_code=True,
        **build_bnb_args(training_args, compute_dtype, skip_lm_head=bnb_skip_lm_head),
    )
    processor = adapter.load_processor(
        processor_source,
        trust_remote_code=True,
        **processor_kwargs,
    )
    return model, processor, adapter, compute_dtype
