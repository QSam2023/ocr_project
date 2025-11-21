# DeepSeek-OCR Model Guide

## 1. Model Architecture

DeepSeek-OCR is a multimodal model designed for Optical Character Recognition (OCR) and document understanding. It combines powerful vision encoders with a large language model (LLM) to process high-resolution images and generate text outputs.

### Key Components:
- **Vision Encoders**: The model utilizes a hybrid vision encoding strategy:
  - **SAM-ViT-B**: Used for detailed local feature extraction.
  - **CLIP-L**: Used for semantic global feature extraction.
- **Projector**: An MLP (Multi-Layer Perceptron) projector maps vision features to the LLM's embedding space.
- **Language Model**: A DeepSeek-V2/V3 based causal language model that takes both text and visual embeddings as input.

### Input Processing (Dynamic Resolution):
The model employs a dynamic resolution strategy to handle images of varying aspect ratios and high details:
1. **Global View**: The entire image is resized to a fixed base size (e.g., 1024x1024) to capture the overall layout.
2. **Local Crops**: The image is dynamically cropped into tiles (e.g., 512x512 or 640x640) based on its aspect ratio. This allows the model to "zoom in" on details without losing resolution.
3. **Tokenization**: Image tiles are converted to visual tokens and interleaved with text tokens. Special tokens (e.g., `<image>`, `<|grounding|>`) are used to structure the input.

## 2. Training Method

The training process involves fine-tuning the model on OCR datasets (image-text pairs).

### Data Preparation:
- **Dataset**: Images and corresponding JSON annotations (containing ground truth text).
- **Splitting**: Data is randomly split into training and testing sets using `data_utils.py`.
- **Preprocessing**: Images are dynamically cropped and normalized. Text prompts (e.g., `<image>\nConvert to markdown`) are prepended to the ground truth.

### Training Loop (`train.py`):
- **Loss Function**: Standard Causal Language Modeling (CLM) loss (Cross-Entropy) on the next token prediction.
- **Masking**: The prompt part of the input is masked so the model only learns to generate the OCR result, not the prompt itself.
- **Optimization**: AdamW optimizer is used.
- **Checkpointing**: Models are saved at the end of each epoch.

## 3. Usage

### Environment Setup
Ensure you have the required dependencies installed (PyTorch, Transformers, etc.).

### Data Splitting
```bash
python data_utils.py --data_root /path/to/ocr_data --output_dir ./data_split
```

### Training
```bash
python train.py \
    --data_root /path/to/ocr_data \
    --train_file ./data_split/train.json \
    --model_path /path/to/deepseek-ocr-model \
    --output_dir ./checkpoints \
    --epochs 3 \
    --batch_size 2
```

### Testing
```bash
python test.py \
    --data_root /path/to/ocr_data \
    --test_file ./data_split/test.json \
    --model_path /path/to/deepseek-ocr-model \
    --output_dir ./results
```

### Experiment Logs
Training and testing logs are saved to `experiment_logs.txt` in the output directory.
