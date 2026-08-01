# ============================================================
# EVALUATION: Base Model vs Fine-Tuned (LoRA) Model
# Run this in the SAME Colab session as training (model is already loaded),
# OR as a fresh session — this script reloads everything from scratch either way.
# ============================================================

# ---- CELL 1: Install dependencies (skip if same session as training) ----
!pip install -q transformers peft bitsandbytes accelerate

# ---- CELL 2: Upload files ----
# Upload: test.json, and the 6 files from your dax_tmdl_lora_final folder
# (adapter_config.json, adapter_model.safetensors, tokenizer.json,
#  tokenizer_config.json, chat_template.jinja)
# Put the adapter files in a folder called dax_tmdl_lora_final in this Colab session.

# ---- CELL 3: Load test data ----
import json

with open("test.json") as f:
    test_data = json.load(f)

print(f"Test examples: {len(test_data)}")
from collections import Counter
print("Category breakdown:", Counter(d["category"] for d in test_data))

# ---- CELL 4: Load BOTH models ----
# Base model = your control group (what it can do with zero fine-tuning)
# Fine-tuned model = base model + your LoRA adapter attached
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"
adapter_path = "./dax_tmdl_lora_final"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto"
)

print("Attaching LoRA adapter for fine-tuned model...")
# PeftModel.from_pretrained loads the base model + your trained adapter together.
# Note: base_model gets "wrapped", not duplicated, so this doesn't double your GPU memory.
finetuned_model = PeftModel.from_pretrained(base_model, adapter_path)

print("Both models ready.")

# ---- CELL 5: Generation helper ----
def generate_response(model, tokenizer, instruction, input_text, max_new_tokens=200):
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{input_text}\n\n"
        f"### Output:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,       # deterministic output - same input always gives same output
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip the prompt back off, keep only what the model generated after "### Output:"
    response = full_text.split("### Output:")[-1].strip()
    return response

# ---- CELL 6: Run both models on all test examples ----
results = []

for i, example in enumerate(test_data):
    print(f"Running example {i+1}/{len(test_data)} ({example['category']})...")

    base_output = generate_response(base_model, tokenizer, example["instruction"], example["input"])
    finetuned_output = generate_response(finetuned_model, tokenizer, example["instruction"], example["input"])

    results.append({
        "category": example["category"],
        "input": example["input"],
        "expected": example["output"],
        "base_output": base_output,
        "finetuned_output": finetuned_output,
    })

with open("eval_results_raw.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done. Raw results saved to eval_results_raw.json")
print("Download this file and bring it back for scoring.")