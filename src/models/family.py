"""Qwen-VL 基础模型族识别工具。"""

import json
from pathlib import Path


QWEN25_FAMILY = "qwen2.5-vl"
QWEN3_FAMILY = "qwen3-vl"


def normalize_ref(value) -> str:
    """将模型路径或模型 id 统一转成小写字符串。"""
    return str(value or "").lower()


def read_model_type(path: str) -> str:
    """从本地 config.json 中读取 model_type，读取失败时返回空字符串。"""
    if not path:
        return ""
    cfg_path = Path(path) / "config.json"
    if not cfg_path.is_file():
        return ""
    try:
        with cfg_path.open(encoding="utf-8") as f:
            return normalize_ref(json.load(f).get("model_type", ""))
    except Exception:
        return ""


def infer_qwen_family(*values) -> str:
    """从路径、模型 id 或 config.json 推断 Qwen-VL 基座家族。"""
    texts = [normalize_ref(value) for value in values if value]
    for value in values:
        model_type = read_model_type(str(value or ""))
        if model_type:
            texts.append(model_type)

    joined = " ".join(texts)
    if "qwen3" in joined or "qwen3_vl" in joined or "qwen3-vl" in joined:
        return QWEN3_FAMILY
    if (
        "qwen2.5" in joined
        or "qwen2_5" in joined
        or "qwen25" in joined
        or "qwen2-vl" in joined
        or "qwen2_vl" in joined
    ):
        return QWEN25_FAMILY
    raise ValueError(
        f"Unsupported Qwen-VL family for values={values!r}. "
        "Expected Qwen2.5-VL or Qwen3-VL references."
    )


def is_qwen25_reference(*values) -> bool:
    """判断模型引用是否属于 Qwen2.5-VL。"""
    return infer_qwen_family(*values) == QWEN25_FAMILY


def is_qwen3_reference(*values) -> bool:
    """判断模型引用是否属于 Qwen3-VL。"""
    return infer_qwen_family(*values) == QWEN3_FAMILY
