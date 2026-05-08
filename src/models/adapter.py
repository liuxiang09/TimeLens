"""模型适配器基类和视觉输入数据结构。"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisionInputs:
    """保存 qwen_vl_utils 输出并生成处理器入参。"""

    images: Any
    videos: Any
    video_kwargs: dict[str, Any] = field(default_factory=dict)
    video_metadata: Any = None

    def processor_kwargs(self):
        """返回可直接传给处理器的视觉相关参数。"""
        kwargs = {
            "images": self.images,
            "videos": self.videos,
            **self.video_kwargs,
        }
        if self.video_metadata is not None:
            kwargs["video_metadata"] = self.video_metadata
        return kwargs


class ModelAdapter:
    """统一封装不同 Qwen-VL 变体的视觉处理差异。"""

    family = ""
    pixel_scale = 1

    def __init__(self, *model_refs):
        """保存用于识别模型变体的路径或编号。"""
        self.model_refs = tuple(ref for ref in model_refs if ref)

    def process_vision_inputs(self, messages):
        """将消息中的视觉内容转换为处理器可接收的输入。"""
        from qwen_vl_utils import process_vision_info

        images, videos, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        return VisionInputs(images=images, videos=videos, video_kwargs=video_kwargs)

    def load_config(self, model_path: str):
        """加载 HuggingFace 配置。"""
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    def load_model(self, model_path: str, **kwargs):
        """加载图文到文本模型。"""
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(model_path, **kwargs)

    def load_processor(self, processor_path: str, **kwargs):
        """加载模型对应的处理器。"""
        from transformers import AutoProcessor

        return AutoProcessor.from_pretrained(processor_path, **kwargs)
