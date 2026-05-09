"""视频时序定位 Bench 推理入口。"""

import argparse
import os
from functools import partial

from src.timelens.data.datasets import DATASET_DICT
from src.models.loader import resolve_processor_source
from src.models.registry import get_adapter
from src.utils.json_io import dump_jsonl
from src.utils.span import extract_time


def load_bench_annos(dataset_name, split="test", bench_root=None):
    """按数据集名称加载 TimeLens-Bench 标注。"""
    dataset_class = DATASET_DICT[dataset_name]
    return dataset_class.load_annos(split=split, bench_root=bench_root)


def parse_args():
    """解析 Bench 推理命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", default="qwen3-vl")
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--bench_root", default=None)
    parser.add_argument("--min_tokens", type=int, default=16)
    parser.add_argument("--total_tokens", type=int, default=3584)
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--fps_max_frames", type=int, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    """运行分片 Bench 推理。"""
    args = parse_args()

    import nncore
    import torch
    from nncore.engine import set_random_seed
    from torch.utils.data import DataLoader

    from src.timelens.data.inference_data import GroundingDatasetInference, collate_fn

    args.seed = set_random_seed(args.seed)
    print(f"Setting random seed to {args.seed}")

    pred_path = f"{args.pred_path}_{args.index}.jsonl"
    print(
        f"Dataset: {args.dataset}({args.split}) | Chunk: {args.chunk} | "
        f"Index: {args.index} | Output Path: {pred_path}"
    )
    if args.device != "auto":
        raise ValueError('Device should be set to "auto" for multi-GPU evaluation.')

    processor_source = resolve_processor_source(args.model_path, args.processor_path)
    args.processor_path = processor_source
    args.format_model_path = processor_source
    adapter = get_adapter(processor_source, args.model_path, args.model_id)

    model = adapter.load_model(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=args.device,
        trust_remote_code=True,
    ).eval()
    processor = adapter.load_processor(
        processor_source,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )

    dataset_class = DATASET_DICT[args.dataset]
    annos = load_bench_annos(args.dataset, split=args.split, bench_root=args.bench_root)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]

    dataset = GroundingDatasetInference(annos, args, adapter=adapter)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=partial(collate_fn, processor=processor, adapter=adapter),
    )

    dumps = []
    for data in nncore.ProgressBar(data_loader):
        inputs = data["inputs"].to("cuda", non_blocking=True)
        annos = data["annos"]

        output_ids = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=512,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        answers = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for anno, answer in zip(annos, answers):
            video_path = anno["video_path"]
            query = anno["query"]
            duration = anno["duration"]
            span = anno["span"]
            timestamps = extract_time(answer)
            if len(timestamps) == 0:
                print("No timestamps extracted, answer might be invalid. Answer:", answer)
                timestamps = [[duration + 10, duration + 20]]

            unit = getattr(dataset_class, "UNIT", 1.0)
            timestamps = [
                [round(start / unit) * unit, round(end / unit) * unit]
                for start, end in timestamps
            ]
            if len(timestamps) > 1:
                timestamps = [timestamps[-1]]
            video_name = os.path.basename(video_path)
            if isinstance(span[0], (list, tuple)):
                span = span[0]
            dumps.append(
                {
                    f"{video_name}>>>{query}>>>{span}": {
                        "timestamps": timestamps,
                        "answer": answer,
                        "query": query,
                        "video_path": video_path,
                        "duration": duration,
                    }
                }
            )
            print(
                f"video_path: {video_path}, query: {query}, duration: {duration}, "
                f"answer: {answer}, extracted timestamps: {timestamps}",
                flush=True,
            )

    dump_jsonl(pred_path, dumps)


if __name__ == "__main__":
    main()
