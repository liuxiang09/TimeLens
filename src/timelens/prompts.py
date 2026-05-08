"""视频时序定位的文本、提示词、时间段和回答格式工具。"""

import re


GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

AUDIO_QUERY_KEYWORDS = {
    "hear",
    "heard",
    "hears",
    "hearing",
    "sound",
    "sounded",
    "sounds",
    "sounding",
    "audio",
}


def grounding_prompt() -> str:
    """返回通用视频时序定位提示词。"""
    return GROUNDING_PROMPT


def parse_query(query):
    """规范化查询文本中的空白字符和结尾标点。"""
    return re.sub(r"\s+", " ", query).strip().strip(".").strip()


def is_audio_related_query(query: str) -> bool:
    """判断查询文本是否明显与音频相关。"""
    words = query.strip("?").lower().split()
    return any(keyword in words for keyword in AUDIO_QUERY_KEYWORDS)


def normalize_spans(span):
    """将单个时间段或时间段列表统一成 [[start, end], ...] 格式。"""
    if isinstance(span, tuple):
        return [list(span)]
    if isinstance(span, list) and len(span) > 0 and isinstance(span[0], (list, tuple)):
        return [list(s) for s in span]
    if isinstance(span, list) and len(span) == 2 and isinstance(span[0], (int, float)):
        return [span]
    raise ValueError(f"Unsupported span format: {span}")


def format_response(spans):
    """将标注时间段格式化为监督训练目标文本。"""
    return (
        "The event happens in "
        + ", ".join([f"{s:.1f} - {e:.1f} seconds" for s, e in spans])
        + "."
    )
