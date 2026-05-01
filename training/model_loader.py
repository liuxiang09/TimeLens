"""Model/config/processor loader for Qwen-VL TimeLens training."""

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from training.model_family import infer_model_family


def _validate_model_path(model_path: str) -> None:
    infer_model_family(model_path)


def get_model_class(model_path: str):
    _validate_model_path(model_path)
    return AutoModelForImageTextToText


def get_config_class(model_path: str):
    _validate_model_path(model_path)
    return AutoConfig


def get_processor_class(model_path: str):
    return AutoProcessor
