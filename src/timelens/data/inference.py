"""TimeLens 推理数据集和批处理整理函数。"""

import copy

from torch.utils.data import Dataset

from src.data.vision import build_processor_inputs
from src.timelens.prompts import grounding_prompt, parse_query
from src.timelens.data.video_content import build_video_content
from src.models.registry import get_adapter


def collate_fn(batch, processor, adapter):
    """将推理样本批量转换为模型输入。"""
    messages = [item["messages"] for item in batch]
    annos = [item["anno"] for item in batch]
    texts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    vision_inputs = adapter.process_vision_inputs(messages)
    inputs = build_processor_inputs(
        processor,
        texts,
        vision_inputs,
        padding=True,
        padding_side="left",
    )
    return {"inputs": inputs, "annos": annos}


class GroundingDatasetInference(Dataset):
    """构造视频时序定位推理阶段的对话消息。"""

    def __init__(self, annos, args, adapter=None):
        """保存标注、运行参数和模型适配器。"""
        super().__init__()
        self.annos = annos
        self.args = args
        self.adapter = adapter or get_adapter(
            getattr(args, "format_model_path", None),
            getattr(args, "processor_path", None),
            getattr(args, "model_path", None),
        )
        self.prompt = grounding_prompt()

    def __len__(self):
        """返回推理样本数量。"""
        return len(self.annos)

    def __getitem__(self, index):
        """返回单条样本的消息和原始标注。"""
        anno = copy.deepcopy(self.annos[index])
        message = {
            "role": "user",
            "content": [
                build_video_content(self.adapter, anno, self.args),
                {"type": "text", "text": self.prompt.format(parse_query(anno["query"]))},
            ],
        }
        return {"messages": [message], "anno": anno}
