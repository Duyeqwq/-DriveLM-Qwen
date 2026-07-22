# DriveLM + Qwen-VL 自动驾驶多模态基座模型微调与评测

本项目把 **DriveLM** 作为自动驾驶图文问答数据与评测任务，把 **Qwen2.5-VL** 作为多模态基座模型，完成数据解析、SFT 格式转换、scene 级训练验证划分、baseline 推理、LoRA/QLoRA 微调和错误分析。

```text
DriveLM 是数据/评测任务
Qwen-VL 是模型基座
本项目负责把二者接起来
```

## 项目目标

- 跑通 DriveLM QA json 与图像数据解析，明确字段、任务类型和样本分布。
- 将 DriveLM 图文问答数据转换为 Qwen-VL SFT 数据格式。
- 按 **scene** 划分训练集和验证集，避免同一场景泄漏到训练和评测。
- 在独立验证集上对比 **原始 Qwen-VL baseline** 与 **LoRA 微调后模型**。
- 扩展单帧/多帧输入、4bit 量化推理和错误可视化分析。

## 数据规模（当前 v1.1）

| 项目 | 规模 |
|------|------|
| scene 数 | 696 |
| QA 样本数 | 377,955 |
| QA json | `v1_1_train_nus.json` (~185MB) |
| 图像 subset | `data/drivelm/nuscenes` (~3.4GB) |
| SFT jsonl | `train_qwen_vl_single_frame.jsonl` (~278MB) |

说明：

- 这是 **DriveLM-nuScenes 官方 train 子集**，不是完整 nuScenes 全量数据。
- 正式对比时，**训练尽量用全量 train split**，评测在 **独立 holdout scene** 上进行。
- 不建议对 37.8 万条全部做 baseline 推理；更有说服力的是：**全量训练 + 独立验证集对比**。

## 目录结构

```text
ad-vlm-project/
├── DriveLM/                  # 原始 DriveLM 仓库
├── Qwen-VL/                  # Qwen 官方代码与微调脚本
├── data/
│   ├── drivelm/              # DriveLM json + images
│   ├── sft/                  # 转换后的 Qwen SFT 数据
│   └── splits/               # scene 级 train/eval 划分
├── scripts/
│   ├── inspect_drivelm.py
│   ├── convert_drivelm_to_qwen.py
│   ├── prepare_scene_split.py
│   ├── run_baseline_infer.py
│   ├── run_baseline_infer_multi_gpu.py
│   ├── eval_predictions.py
│   └── visualize_errors.py
├── experiments/
│   ├── qwen_baseline/
│   ├── qwen_lora_single_frame/
│   ├── qwen_lora_multi_frame/
│   └── qwen_quantized/
└── README.md
```

## 环境准备

建议使用独立环境，避免和 `openvla` 冲突：

```bash
conda create -n ad-vlm python=3.10 -y
conda activate ad-vlm

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U "transformers>=4.49.0" accelerate qwen-vl-utils pillow bitsandbytes peft datasets tqdm
```

验证环境：

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2.5-VL import ok')"
```

## 推荐执行顺序

### Step 0. Clone 代码仓库

```bash
cd /mnt/work/dsg/ad-vlm-project

git clone https://github.com/OpenDriveLab/DriveLM.git DriveLM
git clone https://github.com/QwenLM/Qwen2.5-VL.git Qwen-VL
```

### Step 1. 下载 DriveLM 数据

浏览器申请 Hugging Face 数据集权限：
https://huggingface.co/datasets/OpenDriveLab/DriveLM

服务器登录并下载：

```bash
conda activate ad-vlm
cd /mnt/work/dsg/ad-vlm-project

huggingface-cli login

export HF_HUB_ENABLE_HF_TRANSFER=1
hf download OpenDriveLab/DriveLM \
  --repo-type dataset \
  --include "v1_1_train_nus.json" "drivelm_nus_imgs_train.zip" \
  --local-dir data/drivelm

cd data/drivelm
unzip drivelm_nus_imgs_train.zip
```

### Step 2. 检查原始数据

```bash
cd /mnt/work/dsg/ad-vlm-project

python scripts/inspect_drivelm.py \
  --input data/drivelm/v1_1_train_nus.json \
  --image-root data/drivelm/nuscenes \
  --limit 5
```

重点确认：

- `scene_description`
- `key_frames`
- `image_paths`
- `QA.perception / prediction / planning / behavior`

### Step 3. 转换为 Qwen-VL SFT 格式

单帧（默认 `CAM_FRONT`）：

```bash
python scripts/convert_drivelm_to_qwen.py \
  --input data/drivelm/v1_1_train_nus.json \
  --image-root data/drivelm/nuscenes \
  --output data/sft/train_qwen_vl_single_frame.jsonl \
  --max-images 1
```

多帧（6 相机）后续实验再用：

```bash
python scripts/convert_drivelm_to_qwen.py \
  --input data/drivelm/v1_1_train_nus.json \
  --image-root data/drivelm/nuscenes \
  --output data/sft/train_qwen_vl_multi_frame.jsonl \
  --max-images 6
```

### Step 4. 按 scene 划分 train / eval

不要用 `shuf` 随机抽 QA。应按 scene 划分，避免同一场景泄漏。

```bash
python scripts/prepare_scene_split.py \
  --input data/sft/train_qwen_vl_single_frame.jsonl \
  --output-dir data/splits/scene_split_v1 \
  --eval-ratio 0.1 \
  --eval-subset 10000 \
  --seed 42
```

输出：

```text
data/splits/scene_split_v1/
├── train.jsonl          # 训练集（约 34 万 QA）
├── eval_full.jsonl      # 完整 holdout 验证集（约 3.8 万 QA）
├── eval.jsonl           # 分层抽样 1 万条正式评测集
└── split_manifest.json  # 划分统计与 scene 列表
```

### Step 5. Smoke test（50 条）

先确认推理链路能跑：

```bash
shuf -n 50 data/splits/scene_split_v1/eval.jsonl > data/eval/smoke_50.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline_infer.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --input data/eval/smoke_50.jsonl \
  --output experiments/qwen_baseline/predictions_smoke_50.jsonl \
  --limit 50 \
  --dtype auto \
  --device-map auto
```

### Step 6. Baseline 正式评测

在独立验证集上跑原始 Qwen-VL。推荐用多卡启动器，会自动挑选空闲 GPU：

```bash
python scripts/run_baseline_infer_multi_gpu.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --input data/splits/scene_split_v1/eval.jsonl \
  --output experiments/qwen_baseline/predictions_eval10k.jsonl \
  --limit 10000 \
  --dtype auto \
  --device-map auto
```

手动指定 3 张卡：

```bash
python scripts/run_baseline_infer_multi_gpu.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --input data/splits/scene_split_v1/eval.jsonl \
  --output experiments/qwen_baseline/predictions_eval10k.jsonl \
  --gpus 0,1,2 \
  --limit 10000
```

单卡手动运行仍然支持：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_baseline_infer.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --input data/splits/scene_split_v1/eval.jsonl \
  --output experiments/qwen_baseline/predictions_eval10k.jsonl \
  --limit 10000 \
  --dtype auto \
  --device-map auto
```

如果时间充足，也可以跑完整 holdout：

```bash
python scripts/run_baseline_infer_multi_gpu.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --input data/splits/scene_split_v1/eval_full.jsonl \
  --output experiments/qwen_baseline/predictions_eval_full.jsonl \
  --gpus 0,1,2
```

### Step 7. 评测与错误分析

```bash
python scripts/eval_predictions.py \
  --predictions experiments/qwen_baseline/predictions_eval10k.jsonl \
  --output experiments/qwen_baseline/metrics_eval10k.json

python scripts/visualize_errors.py \
  --predictions experiments/qwen_baseline/predictions_eval10k.jsonl \
  --output experiments/qwen_baseline/error_report_eval10k.html \
  --max-samples 100
```

### Step 8. LoRA 微调

使用 scene split 后的训练集，在 `Qwen-VL/qwen-vl-finetune/` 上做 LoRA/QLoRA。

建议顺序：

1. 先用 `5k~20k` 条确认 loss 能下降
2. 再扩到 `train.jsonl` 全量或大部分
3. 微调后在 **同一个 `eval.jsonl`** 上重新推理并对比

### Step 9. 扩展实验

- 单帧 vs 多帧输入
- 4bit 量化推理
- 视觉 token 压缩
- 分任务错误分析：perception / prediction / planning / behavior

## 实验记录建议

每个实验目录建议保存：

- `config.yaml`
- `train.log`
- `predictions.jsonl`
- `metrics.json`
- `error_report.html`

## 项目方法论

更有说服力的对比方式是：

```text
全量训练集微调
+
独立 scene 验证集评测
+
baseline vs LoRA 对比
+
分任务统计 + 错误可视化
```

不建议：

- 用 `shuf` 从训练集随机抽 eval
- 只跑 50/1000 条就当最终结论
- 对 37.8 万条全部做 baseline 推理

## 简历表述

基于 DriveLM-nuScenes 与 Qwen2.5-VL 构建自动驾驶场景多模态基座模型领域适配流程，完成 DriveLM 图文问答数据解析、SFT 格式转换、scene 级训练验证划分和单帧视觉输入构建；在 3×RTX 3090 服务器上使用 LoRA/QLoRA 完成 Qwen-VL 参数高效微调，并在独立验证集上对比原始模型与微调模型在感知、预测、规划和行为推理类问题上的表现。进一步进行多帧输入、4bit 量化推理和错误样本可视化，分析夜间、遮挡、复杂路口等场景下的失效模式。
