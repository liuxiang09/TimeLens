"""TimeLens-Bench 标注加载工具。"""

from src.timelens.data.datasets import DATASET_DICT


def load_bench_annos(dataset_name, split="test", bench_root=None):
    """按数据集名称加载 TimeLens-Bench 标注。"""
    dataset_class = DATASET_DICT[dataset_name]
    return dataset_class.load_annos(split=split, bench_root=bench_root)
