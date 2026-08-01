# ============================================================
# QLoRA Fine-Tuning: DAX/TMDL Generator (Qwen2.5-Coder-3B-Instruct)
# Run this in Google Colab (T4 GPU runtime: Runtime > Change runtime type > T4 GPU)
# Paste section by section as separate cells, or run as one script.
# ============================================================

# ---- CELL 1: Install dependencies ----
# transformers/peft/bitsandbytes = the core fine-tuning stack
# trl = HuggingFace's SFTTrainer, handles the instruction-tuning loop for us
!pip install -q transformers peft bitsandbytes accelerate trl datasets

# ---- CELL 2: Upload your data files ----
# Upload train.json and val.json to the Colab session (left sidebar > Files > upload)
# or mount Google Drive if you want persistence across sessions.

# ---- CELL 3: Load and format the dataset ----
import json
from datasets import Dataset

with open("train.json") as f:
    train_data = json.load(f)
with open("val.json") as f:
    val_data = json.load(f)

def format_example(ex):
    # This is the prompt template - how we present instruction+input+output to the model.
    # Consistent formatting matters a LOT here: the model learns THIS exact structure,
    # so whatever format you train with is the format you must use at inference time too.
    text = (
        f"### Instruction:\n{ex['instruction']}\n\n"
        f"### Input:\n{ex['input']}\n\n"
        f"### Output:\n{ex['output']}"
    )
    return {"text": text}

train_dataset = Dataset.from_list([format_example(e) for e in train_data])
val_dataset = Dataset.from_list([format_example(e) for e in val_data])

print(f"Train examples: {len(train_dataset)}")
print(f"Val examples: {len(val_dataset)}")
print("\n--- Sample formatted example ---")
print(train_dataset[0]["text"])

# ---- CELL 4: Load base model in 4-bit (this is the "Q" in QLoRA) ----
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

# 4-bit quantization config: this is what lets a 3B model's frozen weights
# fit comfortably in a T4's 16GB VRAM. nf4 = the quantization scheme
# bitsandbytes uses, generally the best accuracy/memory tradeoff available.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Qwen doesn't set this by default

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

# ---- CELL 5: Attach LoRA adapters ----
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

# rank (r) = the "size" of the LoRA adapter matrices. Higher = more capacity to
# learn, but more params to train and more risk of overfitting on small data.
# r=16 is a reasonable starting point for a narrow task like this.
# alpha = scaling factor for the LoRA updates, common convention is alpha = 2*r.
# target_modules = which weight matrices inside the model get LoRA adapters attached.
# These are the attention projection layers - standard choice for LoRA on transformers.
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# ^ This line will show you EXACTLY how few params you're actually training
# vs the full 3B - this is the number that proves LoRA's efficiency claim.

# ---- CELL 6: Training setup ----
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./dax_tmdl_lora",
    num_train_epochs=3,          # start small - 83 examples means each epoch is fast
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,  # effective batch size = 2*4 = 8
    learning_rate=2e-4,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_ratio=0.1,
    bf16=True,
    report_to="none",            # set to "wandb" if you set up W&B tracking
    max_seq_length=512,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# ---- CELL 7: Train ----
trainer.train()

# ---- CELL 8: Save the LoRA adapter ----
# This saves ONLY the adapter (small, ~20-50MB), not the full model.
model.save_pretrained("./dax_tmdl_lora_final")
tokenizer.save_pretrained("./dax_tmdl_lora_final")

print("Training complete. Adapter saved to ./dax_tmdl_lora_final")
print("Download this folder before your Colab session ends (Files > right-click > Download)")
