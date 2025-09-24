"""
This script tests the fine-tuned model by running a sample inference.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# --- Configuration ---
BASE_MODEL_ID = "openai/gpt-oss-20b"  # The base model ID on Hugging Face Hub
ADAPTER_DIR = "gpt-oss-ft"  # Directory where the fine-tuned adapter is saved

def load_model_and_tokenizer(base_model_id: str, adapter_dir: str):
    """
    Load the base model, tokenizer, and fine-tuned adapter.

    Args:
        base_model_id (str): The ID of the base model on the Hugging Face Hub.
        adapter_dir (str): The directory where the fine-tuned adapter is saved.

    Returns:
        tuple: A tuple containing the merged model and the tokenizer.
    """
    # Load the tokenizer for the base model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # Load the base model with specified configurations
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        attn_implementation="eager",  # Use eager attention implementation
        torch_dtype="auto",  # Automatically determine the torch dtype
        use_cache=True,  # Enable caching for faster inference
        device_map="auto",  # Automatically map the model to available devices
    )

    # Load the PEFT model, which combines the base model with the fine-tuned adapter
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)

    # Merge the adapter weights with the base model and unload the adapter
    model = peft_model.merge_and_unload()

    return model, tokenizer

def run_inference(model, tokenizer, messages: list):
    """
    Run inference on the model with the given messages.

    Args:
        model: The fine-tuned model.
        tokenizer: The tokenizer.
        messages (list): A list of messages in the chat format.

    Returns:
        str: The generated response from the model.
    """
    # Apply the chat template to format the input messages and convert to tensors
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    # Disable gradient calculations for inference
    with torch.no_grad():
        # Generate a response from the model
        out = model.generate(inputs, max_new_tokens=128)

    # Decode the generated tokens and return the response
    return tokenizer.batch_decode(out)[0]

def main():
    """
    Main function to run the testing script.
    """
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_ID, ADAPTER_DIR)

    # Define the input messages for the chat-based model
    messages = [
        {"role": "user", "content": "What is 13 + 29? Keep it short."},
    ]

    # Run inference and print the output
    response = run_inference(model, tokenizer, messages)
    print(response)

if __name__ == "__main__":
    main()
