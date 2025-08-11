import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE = "openai/gpt-oss-20b"
ADAPTER_DIR = "gpt-oss-ft"  # your output or Hub repo id

tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE,
    attn_implementation="eager",
    torch_dtype="auto",
    use_cache=True,
    device_map="auto",
)

peft_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model = peft_model.merge_and_unload()

messages = [
    {"role": "user", "content": "What is 13 + 29? Keep it short."},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

with torch.no_grad():
    out = model.generate(inputs, max_new_tokens=128)
print(tokenizer.batch_decode(out)[0])
