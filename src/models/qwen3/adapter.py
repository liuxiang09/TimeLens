"""Qwen3-VL 模型适配器。"""

from src.models.adapter import ModelAdapter, VisionInputs
from src.models.family import QWEN3_FAMILY


class Qwen3Adapter(ModelAdapter):
    """处理 Qwen3-VL 的图像块与视频元数据输入。"""

    family = QWEN3_FAMILY
    pixel_scale = 32 * 32

    def process_vision_inputs(self, messages):
        """按 Qwen3-VL 约定解析视频帧和视频元数据。"""
        from qwen_vl_utils import process_vision_info

        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadata = zip(*videos)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            video_metadata = None
        return VisionInputs(
            images=images,
            videos=videos,
            video_kwargs=video_kwargs,
            video_metadata=video_metadata,
        )
