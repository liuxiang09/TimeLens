"""视觉输入处理器调用辅助函数。"""


def build_processor_inputs(
    processor,
    text,
    vision_inputs,
    padding=False,
    padding_side=None,
    return_tensors="pt",
):
    """把文本和视觉输入对象合并成处理器输入。"""
    kwargs = {
        "text": text,
        "return_tensors": return_tensors,
        "do_resize": False,
        **vision_inputs.processor_kwargs(),
    }
    if padding is not None:
        kwargs["padding"] = padding
    if padding_side is not None:
        kwargs["padding_side"] = padding_side
    return processor(**kwargs)
