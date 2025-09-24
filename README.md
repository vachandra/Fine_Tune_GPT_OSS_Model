# Fine-Tuning a Large Language Model

This repository provides scripts to fine-tune a large language model using Parameter-Efficient Fine-Tuning (PEFT) and the `trl` library.

## Overview

The process involves two main steps:

1.  **Fine-Tuning**: The `finetune.py` script fine-tunes a base model using a provided dataset. It uses LoRA (Low-Rank Adaptation) for efficient tuning.
2.  **Testing**: The `test.py` script loads the fine-tuned model and runs a sample inference to test its performance.

## Setup

1.  **Login to Hugging Face CLI**:
    If you want to push the model to the hub, you need to be logged in.
    ```bash
    huggingface-cli login
    ```

2.  **Install Dependencies**:
    Install the necessary Python libraries using pip.
    ```bash
    pip install "torch" "trl>=0.20.0" "peft>=0.17.0" "transformers>=4.55.0" datasets
    ```

## Usage

### 1. Fine-Tune the Model

Run the `finetune.py` script to start the fine-tuning process. You can customize the training parameters using command-line arguments.

```bash
python finetune.py --bf16
```

**Command-line arguments**:

*   `--model_id`: The base model ID from the Hugging Face Hub (default: `openai/gpt-oss-20b`).
*   `--train_path`: Path to the training data in JSONL format (default: `data/train.jsonl`).
*   `--output_dir`: Directory to save the fine-tuned model checkpoints (default: `gpt-oss-ft`).
*   `--lr`: Learning rate (default: `2e-4`).
*   `--epochs`: Number of training epochs (default: `1`).
*   `--per_device_batch`: Batch size per device (default: `4`).
*   `--grad_accum`: Gradient accumulation steps (default: `4`).
*   `--max_len`: Maximum sequence length (default: `2048`).
*   `--warmup_ratio`: Warmup ratio for the learning rate scheduler (default: `0.03`).
*   `--scheduler`: Learning rate scheduler type (default: `cosine_with_min_lr`).
*   `--min_lr_rate`: Minimum learning rate as a fraction of the initial learning rate (default: `0.1`).
*   `--push_to_hub`: If set, push the model to the Hugging Face Hub.
*   `--hub_repo`: The Hugging Face Hub repository ID to push the model to.
*   `--bf16`: If set, use bfloat16 for training.

### 2. Test the Fine-Tuned Model

After fine-tuning, you can test the model by running the `test.py` script. The script is structured with functions to load the model and run inference.

```bash
python test.py
```

This will load the base model and the fine-tuned adapter, merge them, and then run a sample inference. The script will print the model's output to the console. You can modify the `main` function in `test.py` to change the input messages.
