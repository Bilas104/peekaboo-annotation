import json
import torch
import os
import logging
import sys
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

print("--- STARTING GOLD TRAINING ---")

# CONFIG
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_DIR = os.getcwd()
# POINT TO YOUR NEW HUMAN FILE
DATA_FILE = os.path.join(BASE_DIR, "data/output/human_gold_dataset.json") 
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints_qwen_gold")

# 1. LOAD DATA
if not os.path.exists(DATA_FILE):
    print("You haven't uploaded human_gold_dataset.json yet!")
    exit(1)

with open(DATA_FILE) as f:
    raw = json.load(f)

pairs = []
for item in raw:
    # INPUT: Original Caption
    # OUTPUT: Human Corrected Caption
    original = item['original_caption']
    target = item['human_corrected_caption']
    
    # Chat Format
    text = f"<|im_start|>user\nCorrect this caption: {original}<|im_end|>\n<|im_start|>assistant\n{target}<|im_end|>"
    pairs.append({"text": text})

print(f"Loaded {len(pairs)} Human Gold samples.")
dataset = Dataset.from_list(pairs)

# 2. MODEL
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto", attn_implementation="eager"
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=32, # Higher Rank for better learning on small data
    lora_alpha=64, 
    lora_dropout=0.1, 
    bias="none", 
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)

# 3. TOKENIZE
def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
    labels = tokenized["input_ids"].copy()
    # Mask padding
    labels = [[(t if t != tokenizer.pad_token_id else -100) for t in seq] for seq in labels]
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. TRAIN
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_steps=100,            # Train longer because we have less data
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
    save_strategy="no",
    optim="paged_adamw_8bit",
    neftune_noise_alpha=5,    # <--- MAGIC TRICK for small datasets
    report_to="tensorboard"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print("GOLD TRAINING COMPLETE.")