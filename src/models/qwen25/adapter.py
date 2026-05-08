"""Qwen2.5-VL 模型适配器。"""

from src.models.adapter import ModelAdapter
from src.models.family import QWEN25_FAMILY


class Qwen25Adapter(ModelAdapter):
    """处理 Qwen2.5-VL 原生视频输入。"""

    family = QWEN25_FAMILY
    pixel_scale = 28 * 28
