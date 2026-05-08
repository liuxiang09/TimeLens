"""LoRA 参数选择与 ZeRO-3 权重收集工具。"""

import logging

import torch


def rank0_print(local_rank, *args):
    """仅在主进程或单进程中打印日志。"""
    if local_rank == 0 or local_rank == "0" or local_rank is None or local_rank == -1:
        print(*args)


def find_target_linear_names(
    model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True, local_rank=None
):
    """查找需要注入 LoRA 的线性层或嵌入层模块名。"""
    lora_namespan_exclude = lora_namespan_exclude or []
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(local_rank, f"Found {len(lora_module_names)} lora modules: {lora_module_names}")
    return lora_module_names


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device("cpu")):
    """在 DeepSpeed ZeRO-3 下安全收集单个参数。"""
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if isinstance(device, str):
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logging.warning(
                f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
            )
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    return param.to(device)


def get_peft_state_maybe_zero_3(named_params, bias):
    """收集 LoRA 参数和可选 bias 参数。"""
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    return {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    """收集非 LoRA 参数，可选仅保存可训练参数。"""
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    return {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
