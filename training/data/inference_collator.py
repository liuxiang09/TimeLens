import copy

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

from timelens.dataset.timelens_data import parse_query
from training.model_family import uses_textual_timestamps, video_pixel_scale


GROUNDING_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

GROUNDING_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
) + GROUNDING_PROMPT


def _as_model_refs(model_name):
    if isinstance(model_name, (list, tuple)):
        return tuple(model_name)
    return (model_name,)


def collate_fn(batch, processor, model_name="qwen3-vl"):
    messages = [item["messages"] for item in batch]
    annos = [item["anno"] for item in batch]
    texts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_refs = _as_model_refs(model_name)
    if uses_textual_timestamps(*model_refs):
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if videos is None or len(videos) == 0:
            raise ValueError(
                "Empty videos for Qwen2.5-VL timestamp path. "
                "Please ensure the timestamp processor/config and qwen_vl_utils are aligned."
            )
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            do_resize=False,
        )
    else:
        images, videos, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            padding_side="left",
            return_tensors="pt",
            do_resize=False,
            **video_kwargs,
        )
    return {"inputs": inputs, "annos": annos}


class GroundingDatasetInference(Dataset):
    def __init__(self, annos, args):
        super().__init__()
        self.annos = annos
        self.args = args
        self.model_ref = (
            getattr(args, "format_model_path", None)
            or getattr(args, "processor_path", None)
            or getattr(args, "model_path", "")
            or ""
        )
        self.model_refs = (
            getattr(args, "format_model_path", None),
            getattr(args, "processor_path", None),
            getattr(args, "model_path", None),
        )
        self._uses_textual_timestamps = uses_textual_timestamps(*self.model_refs)
        self._prompt = (
            GROUNDING_PROMPT_TEXT_TIMESTAMP
            if self._uses_textual_timestamps
            else GROUNDING_PROMPT
        )
        self._pixel_scale = video_pixel_scale(*self.model_refs)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])
        video_cfg = {
            "type": "video",
            "video": anno["video_path"],
            "min_pixels": int(self.args.min_tokens * self._pixel_scale),
            "total_pixels": int(self.args.total_tokens * self._pixel_scale),
            "fps": float(self.args.fps),
        }
        if getattr(self.args, "fps_max_frames", None) is not None:
            video_cfg["max_frames"] = int(self.args.fps_max_frames)
        message = {
            "role": "user",
            "content": [
                video_cfg,
                {"type": "text", "text": self._prompt.format(parse_query(anno["query"]))},
            ],
        }
        return {"messages": [message], "anno": anno}
