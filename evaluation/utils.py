# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import copy

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from training.model_family import infer_model_family, uses_textual_timestamps, video_pixel_scale

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# prompt for Qwen2.5-VL based models with interleaved textual timestamps
GROUNDER_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
) + GROUNDER_PROMPT


class GroundingDataset(Dataset):
    def __init__(self, annos, processor, args):
        super().__init__()
        self.annos = annos
        self.processor = processor
        self.args = args
        self.model_refs = (
            getattr(args, "format_model_path", None),
            getattr(args, "processor_path", None),
            args.model_path,
        )
        self.model_family = infer_model_family(*self.model_refs)
        self.uses_textual_timestamps = uses_textual_timestamps(*self.model_refs)
        if self.uses_textual_timestamps:
            # prompt for Qwen2.5-VL based models with interleaved textual timestamps
            self.prompt = GROUNDER_PROMPT_TEXT_TIMESTAMP
        else:
            self.prompt = GROUNDER_PROMPT

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])

        video_path = anno["video_path"]
        query = anno["query"]

        pixel_scale = video_pixel_scale(*self.model_refs)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": self.args.min_tokens * pixel_scale,
                        "total_pixels": self.args.total_tokens * pixel_scale,
                        "fps": self.args.fps,
                    },
                    {"type": "text", "text": self.prompt.format(query)},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if self.uses_textual_timestamps:
            # for Qwen2.5-VL based models with interleaved textual timestamps
            images, videos = process_vision_info(messages, return_video_metadata=True)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
        elif (
            self.model_family == "qwen3-vl"
        ):
            # for Qwen3-VL based models
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Model {self.args.model_path} not supported yet."
            )

        return {"inputs": inputs, "anno": anno}
