"""视频时序定位训练数据过滤推理入口。"""

import argparse
from functools import partial

from src.timelens.data.filtering import load_train_annos
from src.models.loader import resolve_processor_source
from src.models.registry import get_adapter
from src.utils.json_io import dump_jsonl
from src.utils.span import extract_answer, extract_time, iou


def parse_args():
    """解析过滤推理命令行参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gemini_refined_data")
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_id", default="qwen3-vl")
    parser.add_argument("--processor_path", default=None)
    parser.add_argument("--train_jsonl", default=None)
    parser.add_argument("--video_root", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_tokens", type=int, default=64)
    parser.add_argument("--total_tokens", type=int, default=14336)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--fps_max_frames", type=int, default=None)
    return parser.parse_args()


def main():
    """运行分片推理并写出带 IoU 的 JSONL 标注。"""
    args = parse_args()

    import nncore
    import torch
    from nncore.engine import set_random_seed
    from torch.utils.data import DataLoader

    from src.timelens.data.inference import GroundingDatasetInference, collate_fn

    args.seed = set_random_seed(args.seed)
    print(f"Setting random seed to {args.seed}")

    pred_path = f"{args.pred_path}_{args.index}.jsonl"
    print(
        f"Dataset: {args.dataset}({args.split}) | Chunk: {args.chunk} | "
        f"Index: {args.index} | Output Path: {pred_path}"
    )
    if args.device != "auto":
        raise ValueError('Only device="auto" is supported.')

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

    annos = load_train_annos(args.dataset, args.split, args)
    annos.sort(key=lambda x: x["duration"], reverse=True)
    annos = annos[args.index :: args.chunk]

    dataset = GroundingDatasetInference(annos, args, adapter=adapter)
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=10,
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
            repetition_penalty=None,
            max_new_tokens=512,
            use_cache=True,
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
            duration = anno["duration"]
            parsed_answer = extract_answer(answer)
            timestamps = extract_time(parsed_answer)
            if len(timestamps) == 0:
                timestamps = [[duration + 10, duration + 20]]

            gt_span = anno["span"]
            if isinstance(gt_span[0], list):
                gt_span = gt_span[0]
            cur_iou = iou(timestamps[0], gt_span)
            anno.update({"pred": timestamps, "answer": answer, "iou": cur_iou})
            dumps.append(anno)

    dump_jsonl(pred_path, dumps)


if __name__ == "__main__":
    main()
