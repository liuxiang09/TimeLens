"""模型适配器注册与选择。"""

from src.models.family import (
    QWEN25_FAMILY,
    QWEN3_FAMILY,
    infer_qwen_family,
)


def get_adapter(*model_refs):
    """根据模型引用返回 Qwen-VL 模型适配器实例。"""
    from src.models.qwen25.adapter import Qwen25Adapter
    from src.models.qwen3.adapter import Qwen3Adapter

    refs = tuple(ref for ref in model_refs if ref)
    family = infer_qwen_family(*refs)
    if family == QWEN25_FAMILY:
        return Qwen25Adapter(*refs)
    if family == QWEN3_FAMILY:
        return Qwen3Adapter(*refs)
    raise ValueError(f"Unsupported model refs: {model_refs!r}")
