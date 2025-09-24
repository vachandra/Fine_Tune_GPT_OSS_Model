"""
This script fine-tunes a large language model using PEFT and SFTTrainer.
"""
import os
import argparse
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

def parse_args():
    """Parses command-line arguments for the fine-tuning script.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
            - model_id (str): The model ID for the base model from the Hugging Face Hub.
            - train_path (str): The path to the training data in JSONL format.
            - output_dir (str): The directory where model checkpoints will be saved.
            - lr (float): The learning rate for training.
            - epochs (int): The number of training epochs.
            - per_device_batch (int): The batch size per device.
            - grad_accum (int): The number of gradient accumulation steps.
            - max_len (int): The maximum sequence length.
            - warmup_ratio (float): The warmup ratio for the learning rate scheduler.
            - scheduler (str): The type of learning rate scheduler.
            - min_lr_rate (float): The minimum learning rate as a fraction of the initial learning rate.
            - push_to_hub (bool): Whether to push the model to the Hugging Face Hub.
            - hub_repo (str): The repository ID for the Hugging Face Hub.
            - bf16 (bool): Whether to use bfloat16 for training.
    """
    p = argparse.ArgumentParser(description="Fine-tune a model with SFTTrainer")
    p.add_argument("--model_id", type=str, default="openai/gpt-oss-20b", help="Base OSS model on HF Hub.")
    p.add_argument("--train_path", type=str, default="data/train.jsonl", help="Path to JSONL with `messages` records.")
    p.add_argument("--output_dir", type=str, default="gpt-oss-ft", help="Where to save checkpoints.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    p.add_argument("--per_device_batch", type=int, default=4, help="Batch size per device.")
    p.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps.")
    p.add_argument("--max_len", type=int, default=2048, help="Maximum sequence length.")
    p.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio for the learning rate scheduler.")
    p.add_argument("--scheduler", type=str, default="cosine_with_min_lr", help="Learning rate scheduler type.")
    p.add_argument("--min_lr_rate", type=float, default=0.1, help="Minimum learning rate as a fraction of the initial learning rate.")
    p.add_argument("--push_to_hub", action="store_true", help="Push the model to the Hugging Face Hub.")
    p.add_argument("--hub_repo", type=str, default=None, help="The Hugging Face Hub repository ID.")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 for training.")
    return p.parse_args()

def main():
    """
    Main function to run the fine-tuning process.

    This function orchestrates the entire fine-tuning process, including:
    - Parsing command-line arguments.
    - Loading the dataset, tokenizer, and model.
    - Configuring PEFT with LoRA.
    - Setting up the SFTTrainer with training arguments.
    - Running the training process.
    - Saving the final model.
    """
    args = parse_args()

    dataset = load_dataset("json", data_files={"train": args.train_path})["train"]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16 if args.bf16 and torch.cuda.is_available() else "auto",
        use_cache=False,  
        device_map="auto",
    )

    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)


    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules="all-linear",
        target_parameters=[
            "7.mlp.experts.gate_up_proj",
            "7.mlp.experts.down_proj",
            "15.mlp.experts.gate_up_proj",
            "15.mlp.experts.down_proj",
            "23.mlp.experts.gate_up_proj",
            "23.mlp.experts.down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        learning_rate=args.lr,
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        logging_steps=1,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_len,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.scheduler,
        lr_scheduler_kwargs={"min_lr_rate": args.min_lr_rate},
        output_dir=args.output_dir,
        report_to="none",
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()

    print("Done. Model saved to:", args.output_dir)

if __name__ == "__main__":
    main()
