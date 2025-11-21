# DeepSeek-OCR 训练项目

本项目实现了基于 DeepSeek-OCR 模型的微调训练和推理测试流程。

## 环境搭建

本项目推荐使用 `uv` 进行环境管理，以获得更快的依赖安装体验。

### 1. 安装 uv

如果尚未安装 `uv`，请使用以下命令安装：

```bash
pip install uv
```

### 2. 创建并激活虚拟环境

```bash
# 创建虚拟环境
uv venv

# 激活虚拟环境 (Linux/macOS)
source .venv/bin/activate

# 激活虚拟环境 (Windows)
# .venv\Scripts\activate
```

### 3. 安装依赖

```bash
uv pip install -r requirements.txt
```

## 数据准备

请确保数据格式符合 `ocr_data/split/train.json` 和 `ocr_data/split/test.json` 的结构。
- `image`: 图片文件名（相对于 `data_root`）。
- `prompt`: 提示词（通常包含 `<image>` 标签）。
- `ground_truth`: 对应的真实文本标签。

## 模型训练

使用 `train.py` 脚本进行模型微调。

### 参数说明

- `--data_root`: 图片数据的根目录。
- `--train_file`: 训练数据 JSON 文件路径。
- `--model_path`: 预训练模型路径（DeepSeek-OCR 模型文件夹）。
- `--output_dir`: 输出目录，用于保存日志和模型检查点。
- `--epochs`: 训练轮数（默认: 3）。
- `--batch_size`: 批次大小（默认: 2）。
- `--lr`: 学习率（默认: 1e-5）。

### 运行示例

```bash
python train.py \
    --data_root ./ocr_data \
    --train_file ./ocr_data/split/train.json \
    --model_path /path/to/deepseek-ocr-model \
    --output_dir ./output/train_logs \
    --epochs 5 \
    --batch_size 2 \
    --lr 1e-5
```

## 推理与测试

使用 `test.py` 脚本评估模型性能。

### 参数说明

- `--data_root`: 图片数据的根目录。
- `--test_file`: 测试数据 JSON 文件路径。
- `--model_path`: 模型路径（可以是原始模型或训练后的检查点）。
- `--output_dir`: 输出目录，用于保存测试结果。
- `--batch_size`: 批次大小（默认: 2）。
- `--max_new_tokens`: 最大生成 token 数（默认: 1024）。

### 运行示例

```bash
python test.py \
    --data_root ./ocr_data \
    --test_file ./ocr_data/split/test.json \
    --model_path ./output/train_logs/checkpoint-epoch-5 \
    --output_dir ./output/test_results \
    --batch_size 2
```

测试完成后，结果将保存在输出目录下的 `test_results.json` 中，包含平均 Levenshtein 距离和字符错误率 (CER)。

