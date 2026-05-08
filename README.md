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
