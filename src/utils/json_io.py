"""JSON/JSONL 读写工具。"""

import json
from pathlib import Path


def read_json(path):
    """读取 JSON 文件并返回解析后的对象。"""
    with open(path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def write_json(path, data):
    """将对象写入 JSON 文件。"""
    with open(path, "w", encoding="utf-8") as fout:
        json.dump(data, fout, ensure_ascii=False)


def dump_jsonl(path, rows):
    """将多行字典写入 JSONL 文件。"""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl_dict(path):
    """读取每行都是 dict 的 JSONL 文件，并合并为一个字典。"""
    data = {}
    with open(path, "r", encoding="utf-8") as reader:
        for line in reader:
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("Each line in the JSONL file should be a dictionary.")
            data.update(item)
    return data

