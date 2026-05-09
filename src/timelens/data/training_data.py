"""SFT 和 GRPO 使用的视频时序定位 PyTorch 数据集。"""

import copy
from itertools import accumulate
from pathlib import Path

from torch.utils.data import Dataset

from src.data.chatml import preprocess
from src.data.vision import build_processor_inputs
from src.timelens.data.filtering import (
    build_default_filter_args,
    filter_annos,
    load_filtered_annos,
    load_train_annos,
)
from src.timelens.prompts import (
    format_response,
    grounding_prompt,
    normalize_spans,
)
from src.timelens.data.inference_data import build_video_content
from src.models.registry import get_adapter


class GroundingDataset(Dataset):
    """SFT/GRPO 视频时序定位数据集。"""

    def __init__(
        self,
        processor,
        model_args,
        data_args,
        training_args,
        dataset_name: str,
        filter_args=None,
        training_mode: str = "sft",
    ):
        """加载标注、应用过滤规则并初始化模型适配器。"""
        super().__init__()
        self.processor = processor
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.training_mode = training_mode
        self.adapter = get_adapter(
            model_args.processor_path,
            model_args.model_name_or_path,
            model_args.model_id,
        )

        raw_annos = self._load_raw_annos(dataset_name)
        annos = self._apply_basic_filters(raw_annos)
        if filter_args is not None:
            annos = filter_annos(annos, filter_args, data_args, training_args)

        self.annos = annos
        self.raw_length = len(raw_annos)

    def _load_raw_annos(self, dataset_name):
        """根据数据集名称加载原始标注。"""
        if dataset_name in ("gemini_refined_data", "timelens-100k"):
            return load_train_annos(dataset_name, "train", self.data_args)
        if dataset_name == "filtered_hybrid":
            if not self.data_args.raw_anno_path:
                raise ValueError("raw_anno_path is required for filtered_hybrid dataset.")
            if not Path(self.data_args.raw_anno_path).exists():
                raise FileNotFoundError(
                    f"raw_anno_path does not exist: {self.data_args.raw_anno_path}"
                )
            return load_filtered_annos(self.data_args.raw_anno_path)
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def _apply_basic_filters(self, raw_annos):
        """按查询文本长度、视频时长和时间段合法性过滤标注。"""
        annos = []
        for anno in raw_annos:
            num_words = len(anno["query"].split(" "))
            if self.data_args.min_num_words >= 0 and num_words < self.data_args.min_num_words:
                continue
            if self.data_args.max_num_words >= 0 and num_words > self.data_args.max_num_words:
                continue
            if (
                self.data_args.min_video_len >= 0
                and anno.get("duration", float("inf")) < self.data_args.min_video_len
            ):
                continue
            if (
                self.data_args.max_video_len >= 0
                and anno.get("duration", 0) > self.data_args.max_video_len
            ):
                continue
            duration = anno.get("duration")
            spans = normalize_spans(anno["span"])
            if duration and not any(0 <= s <= e <= duration for s, e in spans):
                continue
            anno = dict(anno)
            anno["span"] = spans
            annos.append(anno)
        return annos

    def __len__(self):
        """返回过滤后的数据集长度。"""
        return len(self.annos)

    def __getitem__(self, idx):
        """返回一条 SFT 或 GRPO 样本。"""
        if self.training_mode == "sft":
            return self._getitem_sft(idx)
        if self.training_mode == "grpo":
            return self._getitem_grpo(idx)
        raise ValueError(f"Unsupported training_mode: {self.training_mode}")

    def _build_user_messages(self, anno, include_video_range=False):
        """构造单轮用户消息。"""
        prompt = grounding_prompt()
        return [
            {
                "role": "user",
                "content": [
                    build_video_content(
                        self.adapter,
                        anno,
                        self.data_args,
                        include_video_range=include_video_range,
                    ),
                    {"type": "text", "text": prompt.format(anno["query"])},
                ],
            }
        ]

    def _getitem_sft(self, idx):
        """构造包含 input_ids 和 labels 的 SFT 样本。"""
        anno = copy.deepcopy(self.annos[idx])
        spans = normalize_spans(anno["span"])
        messages = self._build_user_messages(anno)
        vision_inputs = self.adapter.process_vision_inputs(messages)

        messages.append({"role": "assistant", "content": format_response(spans)})
        text = [self.processor.apply_chat_template(messages, tokenize=False).strip()]
        inputs = build_processor_inputs(self.processor, text, vision_inputs, padding=None)
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["labels"] = preprocess(
            inputs["input_ids"],
            text[0],
            self.processor.tokenizer,
            self.model_args.conv_type,
        )
        return inputs

    def _getitem_grpo(self, idx):
        """构造包含提示词、提示词文本和标注的 GRPO 样本。"""
        anno = copy.deepcopy(self.annos[idx])
        messages = self._build_user_messages(anno, include_video_range=True)
        text = [
            self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        ]
        vision_inputs = self.adapter.process_vision_inputs(messages)

        inputs = build_processor_inputs(self.processor, text, vision_inputs, padding=None)
        inputs["input_ids"] = inputs["input_ids"][0]
        inputs["prompt"] = messages
        inputs["prompt_text"] = text[0]
        inputs["anno"] = anno
        return inputs


class HybridDataset(Dataset):
    """视频时序定位多数据集封装器。"""

    def __init__(
        self,
        processor,
        model_config,
        model_args,
        data_args,
        training_args,
        training_mode="sft",
    ):
        """根据数据参数中的数据集配置构造并拼接多个时序定位数据集。"""
        super().__init__()
        if not data_args.datasets:
            raise ValueError("data_args.datasets is required.")

        dataset_names = [name.strip() for name in data_args.datasets.split(",") if name.strip()]
        datasets = []
        for name in dataset_names:
            if name in ("gemini_refined_data", "timelens-100k"):
                filter_args = build_default_filter_args(data_args.target_size)
                dataset_name = name
            elif name == "filtered_hybrid":
                filter_args = build_default_filter_args(data_args.target_size)
                dataset_name = "filtered_hybrid"
            else:
                raise ValueError(
                    f"Unsupported dataset name: {name}. "
                    "Supported: gemini_refined_data, timelens-100k, filtered_hybrid."
                )

            datasets.append(
                GroundingDataset(
                    processor=processor,
                    model_args=model_args,
                    data_args=data_args,
                    training_args=training_args,
                    dataset_name=dataset_name,
                    filter_args=filter_args,
                    training_mode=training_mode,
                )
            )

        cum_length = [0] + list(accumulate([len(d) for d in datasets]))
        self.idx_ranges = [[cum_length[i], cum_length[i + 1]] for i in range(len(cum_length) - 1)]
        self.datasets = datasets

    def __len__(self):
        """返回拼接后的数据集长度。"""
        return self.idx_ranges[-1][-1]

    def __getitem__(self, idx):
        """将全局索引路由到对应的子数据集。"""
        for (start, end), dataset in zip(self.idx_ranges, self.datasets):
            if start <= idx < end:
                return dataset[idx - start]
        raise IndexError(f"Index out of range: {idx}")
