"""HuggingFace 模型、配置和处理器加载薄封装。"""

from src.models.registry import get_adapter


def validate_model_refs(*model_refs) -> None:
    """通过适配器注册表校验模型引用是否受支持。"""
    get_adapter(*model_refs)


def get_model_class(*model_refs):
    """返回当前项目统一使用的模型类。"""
    from transformers import AutoModelForImageTextToText

    validate_model_refs(*model_refs)
    return AutoModelForImageTextToText


def get_config_class(*model_refs):
    """返回当前项目统一使用的配置类。"""
    from transformers import AutoConfig

    validate_model_refs(*model_refs)
    return AutoConfig


def get_processor_class(*model_refs):
    """返回当前项目统一使用的处理器类。"""
    from transformers import AutoProcessor

    validate_model_refs(*model_refs)
    return AutoProcessor


def resolve_processor_source(model_path: str, processor_path=None) -> str:
    """优先使用显式处理器路径，否则复用模型路径。"""
    return processor_path or model_path
