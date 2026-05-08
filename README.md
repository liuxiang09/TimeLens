# TimeLens Core

基于 Qwen2.5-VL / Qwen3-VL 的 TimeLens 视频时序定位训练、过滤、GRPO 和评测代码。

## 安装

```bash
conda create -n timelens python=3.11 -y
conda activate timelens

pip install -r requirements.txt -f https://download.pytorch.org/whl/cu124
pip install -r requirements_train.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
```

## 常用命令

### Qwen3-VL SFT

```bash
bash scripts/qwen3/sft.sh \
  --model_id "qwen3-vl-8b" \
  --model_path "/path/to/Qwen3-VL-8B-Instruct" \
  --train_jsonl "/path/to/TimeLens-100K/timelens-100k.jsonl" \
  --video_root "/path/to/TimeLens-100K/videos"
```

### Qwen3-VL 数据过滤

```bash
bash scripts/temporal_grounding/filter.sh \
  --model_id "qwen3-vl-8b" \
  --model_path "output/temporal_grounding/qwen3/sft/YOUR_SFT_RUN_DIR" \
  --dataset "gemini_refined_data"
```

### Qwen3-VL GRPO

```bash
bash scripts/qwen3/grpo.sh \
  --model_id "qwen3-vl-8b" \
  --model_path "output/temporal_grounding/qwen3/sft/YOUR_SFT_RUN_DIR" \
  --raw_anno_path "output/temporal_grounding/filter-data/qwen3-vl-8b/YOUR_FILTER_RUN_DIR/gemini_refined_data.jsonl"
```

### Qwen2.5-VL SFT

```bash
bash scripts/qwen25/sft.sh \
  --model_id "qwen2.5-vl-7b" \
  --model_path "/path/to/Qwen2.5-VL-7B-Instruct" \
  --train_jsonl "/path/to/TimeLens-100K/timelens-100k.jsonl" \
  --video_root "/path/to/TimeLens-100K/videos"
```

### Qwen2.5-VL GRPO

```bash
bash scripts/qwen25/grpo.sh \
  --model_id "qwen2.5-vl-7b" \
  --model_path "output/temporal_grounding/qwen25/sft/YOUR_SFT_RUN_DIR" \
  --raw_anno_path "output/temporal_grounding/filter-data/qwen2.5-vl-7b/YOUR_FILTER_RUN_DIR/gemini_refined_data.jsonl"
```

### TimeLens-Bench 评测

```bash
CUDA_VISIBLE_DEVICES="0,1" \
bench_root="/path/to/TimeLens-Bench" \
model_id="qwen3-vl-8b" \
model_path="output/temporal_grounding/qwen3/grpo/YOUR_GRPO_RUN_DIR" \
bash scripts/temporal_grounding/eval.sh
```

也可以直接调用 Python 入口：

```bash
python -m src.timelens.train.sft --help
python -m src.timelens.train.grpo --help
python -m src.timelens.data.filter_data --help
python -m src.timelens.eval.eval_bench --help
python -m src.timelens.eval.metrics --help
```

## 当前代码结构

```text
src/
  models/       # Qwen2.5-VL / Qwen3-VL 模型适配、模型族识别和 processor 加载
  timelens/     # TimeLens prompt、数据、过滤、reward、训练、推理和评测
  training/     # 通用训练参数、LoRA、保存逻辑、SFT/GRPO Trainer
  data/         # 通用 ChatML、collator、vision processor 输入辅助
  utils/        # JSON、时间戳解析、temporal IoU 等工具

scripts/
  qwen25/       # Qwen2.5-VL SFT / GRPO 脚本
  qwen3/        # Qwen3-VL SFT / GRPO 脚本
  temporal_grounding/  # TimeLens 数据过滤和 Bench 评测脚本
```
