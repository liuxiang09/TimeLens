"""TimeLens-100K 和 TimeLens-Bench 标注加载器。"""

import json
import os
from pathlib import Path

from src.timelens.prompts import parse_query


def _resolve_required_path(value, env_name, description):
    """从显式参数或环境变量中解析必需路径。"""
    resolved = value or os.environ.get(env_name)
    if not resolved:
        raise ValueError(
            f"{description} is required. Pass the CLI argument or set {env_name}."
        )
    return Path(resolved).expanduser()


def _resolve_bench_root(bench_root=None):
    """解析 TimeLens-Bench 根目录。"""
    return _resolve_required_path(bench_root, "TIMELENS_BENCH_ROOT", "bench_root")


class ActivitynetTimeLensDataset:
    """ActivityNet-TimeLens 评测集加载器。"""

    ANNO_FILE = "activitynet-timelens.json"
    VIDEO_SUBDIR = "activitynet"
    DATASET_SOURCE = "ActivityNet-TimeLens"

    @classmethod
    def load_annos(cls, split="test", bench_root=None, **kwargs):
        """加载评测标注并补全视频路径。"""
        if split != "test":
            raise ValueError(f"Invalid split: {split}")
        root = _resolve_bench_root(bench_root)
        anno_path = root / cls.ANNO_FILE
        video_root = root / "videos" / cls.VIDEO_SUBDIR

        with anno_path.open("r", encoding="utf-8") as f:
            raw_annos = json.load(f)

        annos = []
        for vid, raw_anno in raw_annos.items():
            video_path = str(video_root / f"{vid}.mp4")
            for span, query in zip(raw_anno["spans"], raw_anno["queries"]):
                annos.append(
                    {
                        "source": cls.DATASET_SOURCE,
                        "data_type": "grounding",
                        "video_path": video_path,
                        "duration": raw_anno["duration"],
                        "query": parse_query(query),
                        "span": [span],
                    }
                )
        return annos


class QVHighlightsTimeLensDataset(ActivitynetTimeLensDataset):
    """QVHighlights-TimeLens 评测集加载器。"""

    ANNO_FILE = "qvhighlights-timelens.json"
    VIDEO_SUBDIR = "qvhighlights"
    DATASET_SOURCE = "QVHighlights-TimeLens"


class CharadesTimeLensDataset(ActivitynetTimeLensDataset):
    """Charades-TimeLens 评测集加载器。"""

    ANNO_FILE = "charades-timelens.json"
    VIDEO_SUBDIR = "charades"
    DATASET_SOURCE = "Charades-TimeLens"


class TimeLens100KDataset:
    """TimeLens-100K 训练集加载器。"""

    @classmethod
    def load_annos(cls, split="train", train_jsonl=None, video_root=None, **kwargs):
        """加载训练标注并展开为事件级样本。"""
        if split != "train":
            raise ValueError(f"Invalid split: {split}")
        anno_path = _resolve_required_path(
            train_jsonl,
            "TIMELENS_100K_JSONL",
            "train_jsonl",
        )
        video_root = _resolve_required_path(
            video_root,
            "TIMELENS_100K_VIDEO_ROOT",
            "video_root",
        )

        raw_rows = []
        with anno_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw_rows.append(json.loads(line))

        annos = []
        for raw_anno in raw_rows:
            video_path = str(video_root / raw_anno["video_path"])
            for event in raw_anno["events"]:
                annos.append(
                    {
                        "source": raw_anno["source"],
                        "data_type": "grounding",
                        "video_path": video_path,
                        "duration": raw_anno["duration"],
                        "query": parse_query(event["query"]),
                        "span": event["span"],
                    }
                )
        return annos


DATASET_DICT = {
    "activitynet-timelens": ActivitynetTimeLensDataset,
    "qvhighlights-timelens": QVHighlightsTimeLensDataset,
    "charades-timelens": CharadesTimeLensDataset,
    "timelens-100k": TimeLens100KDataset,
}
