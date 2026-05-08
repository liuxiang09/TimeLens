"""时间段解析和时序交并比工具。"""

import re


def extract_answer(content):
    """从 GRPO 的 <answer> 标签中抽取最终回答。"""
    format_pattern = r"<think>.*?</think>\s*<answer>(.*?)</answer>"
    match = re.match(format_pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    if any(tag in content for tag in ("<think>", "</think>", "<answer>", "</answer>")):
        return content
    return content


def iou(a, b):
    """计算两个 [start, end] 时间段的时序交并比。"""
    max0 = max(a[0], b[0])
    min0 = min(a[0], b[0])
    max1 = max(a[1], b[1])
    min1 = min(a[1], b[1])
    denom = max1 - min0
    if denom <= 0:
        return 0.0
    return max(min1 - max0, 0) / denom


def extract_time(paragraph):
    """从文本中抽取成对时间戳，返回 [(start, end), ...] 秒级时间段。"""
    paragraph = paragraph.lower().replace("to", "-")
    timestamps = []

    time_regex = re.compile(
        r"\b(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?|\d{1,2}:\d{2}(?:\.\d+)?)\b"
    )
    time_matches = re.findall(time_regex, paragraph)
    time_matches = time_matches[: len(time_matches) // 2 * 2]

    if time_matches:
        converted = []
        for raw_time in time_matches:
            parts = raw_time.split(":")
            if len(parts) == 3:
                h, m = map(int, parts[:2])
                s = float(parts[2])
                time_in_sec = h * 3600 + m * 60 + s
            else:
                m = int(parts[0])
                s = float(parts[1])
                time_in_sec = m * 60 + s
            converted.append(float(time_in_sec))
        timestamps = [
            (converted[i], converted[i + 1]) for i in range(0, len(converted), 2)
        ]

    if len(timestamps) == 0:
        patterns = [
            r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)",
            r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)",
        ]
        for pattern in patterns:
            time_matches = re.findall(pattern, paragraph)
            if time_matches:
                timestamps = [(float(start), float(end)) for start, end in time_matches]
                break

    if len(timestamps) == 0:
        time_regex = re.compile(r"\b(\d+\.\d+|\d+)\b")
        time_matches = re.findall(time_regex, paragraph)
        time_matches = time_matches[: len(time_matches) // 2 * 2]
        timestamps = [
            (float(time_matches[i]), float(time_matches[i + 1]))
            for i in range(0, len(time_matches), 2)
        ]

    return [(start, end) for start, end in timestamps]
