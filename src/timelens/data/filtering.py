"""视频时序定位标注加载、过滤和采样工具。"""

import random

import numpy as np

from src.timelens.data.datasets import TimeLens100KDataset
from src.timelens.prompts import is_audio_related_query, parse_query


def build_default_filter_args(target_size: int):
    """构造默认的视频时长分桶采样目标。"""
    ranges = [(i, i + 30) for i in range(0, 240, 30)] + [(240, float("inf"))]
    per_range = int(target_size / len(ranges))
    return {
        "filter_range": ranges,
        "filter_target_size": [per_range] * len(ranges),
    }


def load_filtered_annos(path: str):
    """读取过滤推理阶段生成的标注。"""
    import nncore

    loaded = nncore.load(path)
    if isinstance(loaded, dict):
        loaded = [loaded]
    if loaded is None:
        return []
    annos = []
    for raw in loaded:
        if "source" not in raw or "query" not in raw:
            continue
        annos.append(
            {
                "source": raw["source"],
                "data_type": raw.get("data_type", "grounding"),
                "video_path": raw["video_path"],
                "duration": raw["duration"],
                "query": parse_query(raw["query"]),
                "span": raw["span"],
                "iou": raw.get("iou"),
                "pred": raw.get("pred"),
                "answer": raw.get("answer"),
            }
        )
    return annos


def load_train_annos(dataset_names: str, split: str, data_args=None):
    """根据数据集名称加载训练标注。"""
    if split != "train":
        raise ValueError("Only train split is supported in filtering/training stage.")
    annos = []
    for dataset in dataset_names.split(","):
        dataset = dataset.strip()
        train_kwargs = {
            "train_jsonl": getattr(data_args, "train_jsonl", None),
            "video_root": getattr(data_args, "video_root", None),
        }
        if dataset == "gemini_refined_data":
            annos.extend(
                [
                    anno
                    for anno in TimeLens100KDataset.load_annos(
                        split="train", **train_kwargs
                    )
                    if not is_audio_related_query(anno["query"])
                ]
            )
        elif dataset == "timelens-100k":
            annos.extend(
                TimeLens100KDataset.load_annos(split="train", **train_kwargs)
            )
        else:
            raise ValueError(f"Unsupported dataset for filtering: {dataset}")
    return annos


def filter_annos(annos, filter_args, data_args, training_args):
    """按视频时长分桶和可选高斯权重采样标注。"""
    unique_videos = filter_args.get("unique_videos", False)
    if unique_videos:
        seen = set()
        uniq = []
        for anno in annos:
            vpath = anno["video_path"]
            if vpath in seen:
                continue
            seen.add(vpath)
            uniq.append(anno)
        annos = uniq

    filter_ratio = filter_args.get("filter_ratio")
    filter_target_size = filter_args.get("filter_target_size")
    if filter_ratio is None and filter_target_size is None:
        return annos

    gaussian_filter_mean = getattr(data_args, "gaussian_filter_mean", None)
    gaussian_filter_std = getattr(data_args, "gaussian_filter_std", None)
    if (gaussian_filter_mean is None) != (gaussian_filter_std is None):
        raise ValueError(
            "gaussian_filter_mean and gaussian_filter_std should be provided together."
        )
    if gaussian_filter_mean is not None and not annos:
        return annos
    if gaussian_filter_mean is not None and "iou" not in annos[0]:
        raise ValueError("Gaussian filtering requires 'iou' in annotations.")

    seed = getattr(training_args, "seed", 42)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    buckets = {duration_range: [] for duration_range in filter_args["filter_range"]}
    kept_indices = []
    for idx, anno in enumerate(annos):
        matched = False
        for duration_range in buckets:
            min_duration, max_duration = duration_range
            if min_duration <= anno["duration"] <= max_duration:
                buckets[duration_range].append(idx)
                matched = True
                break
        if not matched:
            kept_indices.append(idx)

    for i, indices in enumerate(buckets.values()):
        if len(indices) == 0:
            continue
        num_to_select = (
            int(len(indices) * filter_ratio[i])
            if filter_ratio is not None
            else int(filter_target_size[i])
        )
        num_to_select = min(num_to_select, len(indices))

        if gaussian_filter_mean is not None:
            iou_list = np.array([annos[idx]["iou"] for idx in indices], dtype=np.float64)
            weights = np.exp(
                -0.5 * ((iou_list - gaussian_filter_mean) / gaussian_filter_std) ** 2
            )
            if getattr(data_args, "fixed_gaussian_sampling", False):
                num_bins = 20
                counts, bin_edges = np.histogram(iou_list, bins=num_bins, range=(0, 1))
                bin_indices = np.digitize(iou_list, bins=bin_edges)
                bin_indices = np.clip(bin_indices, 1, num_bins) - 1
                inverse_density = 1.0 / (counts + 1e-6)
                weights *= inverse_density[bin_indices]
            weights = weights / weights.sum()
            selected_indices = rng.choice(
                indices, size=num_to_select, replace=False, p=weights
            ).tolist()
        else:
            selected_indices = py_rng.sample(indices, num_to_select)
        kept_indices.extend(selected_indices)

    kept_indices = set(kept_indices)
    return [annos[i] for i in range(len(annos)) if i in kept_indices]
