"""TimeLens 视频输入字段构造。"""


def build_video_content(adapter, anno, data_args, include_video_range=False):
    """根据标注和数据参数构造对话模板中的视频字段。"""
    content = {
        "type": "video",
        "video": anno["video_path"],
        "min_pixels": int(data_args.min_tokens * adapter.pixel_scale),
        "total_pixels": int(data_args.total_tokens * adapter.pixel_scale),
        "fps": float(data_args.fps),
    }
    if include_video_range:
        content["video_start"] = anno.get("video_start")
        content["video_end"] = anno.get("video_end")
    if getattr(data_args, "fps_max_frames", None) is not None:
        content["max_frames"] = int(data_args.fps_max_frames)
    return content
