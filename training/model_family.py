import json
from pathlib import Path


QWEN25_FAMILY = "qwen2.5-vl"
QWEN3_FAMILY = "qwen3-vl"


def _normalize(value) -> str:
    return str(value or "").lower()


def _read_model_type(path: str) -> str:
    if not path:
        return ""
    cfg_path = Path(path) / "config.json"
    if not cfg_path.is_file():
        return ""
    try:
        with cfg_path.open(encoding="utf-8") as f:
            return _normalize(json.load(f).get("model_type", ""))
    except Exception:
        return ""


def uses_textual_timestamps(*values) -> bool:
    """Qwen2.5-VL based TimeLens models use textual timestamp encoding."""
    return infer_model_family(*values) == QWEN25_FAMILY


def resolve_processor_source(model_path: str, processor_path=None) -> str:
    if processor_path:
        return processor_path
    return model_path


def infer_model_family(*values) -> str:
    """Infer the Qwen-VL model family from paths, ids, or local config.json files."""
    texts = [_normalize(value) for value in values if value]
    for value in values:
        model_type = _read_model_type(str(value or ""))
        if model_type:
            texts.append(model_type)

    joined = " ".join(texts)
    if (
        "qwen3" in joined
        or "qwen3_vl" in joined
        or "qwen3-vl" in joined
        or "timelens-4b" in joined
        or "timelens-8b" in joined
    ):
        return QWEN3_FAMILY
    if (
        "qwen2.5" in joined
        or "qwen2_5" in joined
        or "qwen25" in joined
        or "qwen2-vl" in joined
        or "qwen2_vl" in joined
        or "timelens-7b" in joined
    ):
        return QWEN25_FAMILY
    raise ValueError(
        f"Unsupported model family for values={values!r}. "
        "Expected Qwen3-VL family or Qwen2.5-VL family."
    )


def is_qwen25_family(*values) -> bool:
    return infer_model_family(*values) == QWEN25_FAMILY


def is_qwen3_family(*values) -> bool:
    return infer_model_family(*values) == QWEN3_FAMILY


def video_pixel_scale(*values) -> int:
    family = infer_model_family(*values)
    if family == QWEN25_FAMILY:
        return 28 * 28
    return 32 * 32
