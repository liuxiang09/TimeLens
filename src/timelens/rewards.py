"""TimeLens 视频时序定位 GRPO 奖励函数。"""

import re

from src.utils.span import extract_answer, extract_time, iou


def format_reward(completions, **kwargs):
    """检查模型输出是否符合 <think>...</think><answer>...</answer> 格式。"""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content) for content in completion_contents]

    for i, match in enumerate(matches):
        if not match:
            print(f"Completion {i} does not match the required format: {completion_contents[i]}")

    return [1.0 if match else 0.0 for match in matches]


def tiou_reward(prompts, completions, completion_ids, anno, prompt_text, **kwargs):
    """根据预测时间段和标注时间段的时序交并比计算奖励。"""
    pattern = r'<\|(video_pad|image_pad|vision_start|vision_end)\|>'
    prompt_text = [re.sub(pattern, "", text) for text in prompt_text]

    completions = [completion[0]["content"] for completion in completions]
    answers = [extract_answer(completion) for completion in completions]
    timestamps_list = [extract_time(answer) for answer in answers]

    rewards = []
    for i, timestamps in enumerate(timestamps_list):
        gt = anno[i]["span"]
        if isinstance(gt[0], list):
            gt = gt[0]

        pred = answers[i]

        if len(timestamps) == 0:
            print(f"Timestamp extraction failed: pred={pred}, IoU will be 0")
            rewards.append(0)
        elif timestamps[0][0] >= timestamps[0][1]:
            print(f"Warning: Invalid timestamp in prediction '{pred}', IoU will be 0")
            rewards.append(0)
        else:
            if len(timestamps) > 1:
                print(f"Warning: Multiple timestamps for '{pred}', using first: {timestamps[0]}")
            rewards.append(iou(gt, timestamps[0]))
            print(
                f"prompt: {prompt_text[i]}, completion: {completions[i]}, "
                f"answer: {pred}, gt: {gt}, tIoU: {rewards[i]}"
            )

    return rewards


REWARD_FUNCS_DICT = {
    "tiou": tiou_reward,
    "format": format_reward,
}


def load_reward_funcs(reward_func_names):
    """根据逗号分隔的名称加载奖励函数列表。"""
    return [
        REWARD_FUNCS_DICT[func_name.strip()]
        for func_name in reward_func_names.split(",")
    ]
