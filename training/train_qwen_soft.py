"""
Qwen Soft Training - Ultra Conservative for Small Datasets
-----------------------------------------------------------
- Very low learning rate
- Minimal LoRA rank
- More epochs with early stopping
- Gradient clipping
- Clean start (delete old checkpoints first)
"""

import json
import torch
import os
import logging
import sys
import random
import shutil
from datetime import datetime
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)

# =============================================================================
# CONFIGURATION - ULTRA CONSERVATIVE
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
BASE_DIR = os.getcwd()
DATA_FILE = os.path.join(BASE_DIR, "data/output/human_gold_dataset.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "checkpoints_qwen_soft")

# SOFT hyperparameters for 115 samples
BATCH_SIZE = 1                # Smaller batch
GRADIENT_ACCUMULATION = 8     # Effective batch = 8
LEARNING_RATE = 1e-5          # Very low LR
NUM_EPOCHS = 10               # More epochs, rely on early stopping
MAX_LENGTH = 128              # Short for captions
LORA_R = 4                    # Minimal rank - prevents overfitting
LORA_ALPHA = 8                # 2x rank
LORA_DROPOUT = 0.1            # Some dropout
WARMUP_RATIO = 0.15           # More warmup
MAX_GRAD_NORM = 0.5           # Aggressive gradient clipping

# =============================================================================
# CLEAN START - DELETE OLD CHECKPOINTS
# =============================================================================

checkpoints_to_delete = [
    "checkpoints_qwen_soft",
    "checkpoints_qwen_fixed", 
    "checkpoints_qwen_simple",
    "checkpoints_qwen_gold",
    "checkpoints_qwen"
]

print("="*80)
print("CLEANING OLD CHECKPOINTS")
print("="*80)

for ckpt_dir in checkpoints_to_delete:
    ckpt_path = os.path.join(BASE_DIR, ckpt_dir)
    if os.path.exists(ckpt_path):
        print(f"Deleting: {ckpt_path}")
        shutil.rmtree(ckpt_path)
    else:
        print(f"Not found (skip): {ckpt_path}")

print("Cleanup complete.\n")

# =============================================================================
# LOGGING
# =============================================================================

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/training_soft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("     QWEN SOFT TRAINING - ULTRA CONSERVATIVE")
logger.info("="*80)
logger.info(f"Learning Rate: {LEARNING_RATE}")
logger.info(f"LoRA Rank: {LORA_R}")
logger.info(f"Max Grad Norm: {MAX_GRAD_NORM}")
logger.info("="*80)

# =============================================================================
# LOAD DATA
# =============================================================================

if not os.path.exists(DATA_FILE):
    logger.error(f"Data file not found: {DATA_FILE}")
    exit(1)

with open(DATA_FILE) as f:
    raw_data = json.load(f)

logger.info(f"Loaded {len(raw_data)} examples")

# =============================================================================
# FORMAT DATA - SIMPLE COMPLETION
# =============================================================================

PROMPT_TEMPLATE = "### Input:\n{original}\n\n### Corrected:\n"

def format_example(item):
    original = item['original_caption'].strip()
    corrected = item['human_corrected_caption'].strip()
    
    prompt = PROMPT_TEMPLATE.format(original=original)
    full_text = prompt + corrected
    
    return {
        "prompt": prompt,
        "completion": corrected,
        "full_text": full_text
    }

formatted_data = [format_example(item) for item in raw_data]

# Show examples
logger.info("\nFormat examples:")
for i in range(2):
    logger.info(f"\n--- Example {i+1} ---")
    logger.info(f"Full text:\n{formatted_data[i]['full_text']}")

# Shuffle and split
random.seed(42)
random.shuffle(formatted_data)

split_idx = int(len(formatted_data) * 0.9)
train_data = formatted_data[:split_idx]
val_data = formatted_data[split_idx:]

logger.info(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")

# =============================================================================
# LOAD MODEL - FRESH
# =============================================================================

logger.info(f"\nLoading FRESH model: {MODEL_ID}")

# Clear CUDA cache
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    quantization_config=bnb_config, 
    device_map="auto",
    attn_implementation="eager",
    trust_remote_code=True
)

# =============================================================================
# VERIFY BASE MODEL WORKS
# =============================================================================

logger.info("\nTesting BASE model before any training...")

test_prompt = "### Input:\na woman with purple hair\n\n### Corrected:\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

base_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
logger.info(f"Base model output: {base_output}")

# Check for garbage
if "![" in base_output or "!@#" in base_output:
    logger.error("BASE MODEL IS ALREADY PRODUCING GARBAGE!")
    logger.error("This is a fundamental issue with the model or environment.")
else:
    logger.info("Base model output looks clean. Proceeding with training.")

# =============================================================================
# PREPARE FOR TRAINING
# =============================================================================

model = prepare_model_for_kbit_training(model)

# =============================================================================
# CONFIGURE LORA - MINIMAL
# =============================================================================

peft_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]  # Only Q and V - minimal
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize_function(examples):
    all_input_ids = []
    all_attention_mask = []
    all_labels = []
    
    for prompt, completion, full_text in zip(examples["prompt"], examples["completion"], examples["full_text"]):
        full_tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        
        prompt_tokens = tokenizer(prompt, add_special_tokens=True)
        prompt_len = len(prompt_tokens["input_ids"])
        
        labels = full_tokens["input_ids"].copy()
        
        # Mask prompt
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        
        # Mask padding
        for i in range(len(labels)):
            if labels[i] == tokenizer.pad_token_id:
                labels[i] = -100
        
        all_input_ids.append(full_tokens["input_ids"])
        all_attention_mask.append(full_tokens["attention_mask"])
        all_labels.append(labels)
    
    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_mask,
        "labels": all_labels
    }

logger.info("\nTokenizing...")

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

train_tokenized = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
val_tokenized = val_dataset.map(tokenize_function, batched=True, remove_columns=val_dataset.column_names)

# =============================================================================
# DATA COLLATOR
# =============================================================================

class SimpleCollator:
    def __call__(self, features):
        return {
            "input_ids": torch.tensor([f["input_ids"] for f in features]),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features]),
            "labels": torch.tensor([f["labels"] for f in features]),
        }

# =============================================================================
# TRAINING - SOFT CONFIGURATION
# =============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Batch
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    
    # Duration
    num_train_epochs=NUM_EPOCHS,
    
    # Learning rate - SOFT
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,
    max_grad_norm=MAX_GRAD_NORM,  # Aggressive clipping
    
    # Precision
    bf16=True,
    
    # Logging
    logging_steps=5,
    
    # Evaluation
    eval_strategy="steps",
    eval_steps=25,
    
    # Saving
    save_strategy="steps",
    save_steps=25,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    # Optimizer
    optim="paged_adamw_8bit",
    
    # Other
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=False,  # Disable to avoid issues
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    data_collator=SimpleCollator(),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

logger.info("\n" + "="*80)
logger.info("STARTING SOFT TRAINING")
logger.info("="*80)

trainer.train()

# =============================================================================
# SAVE
# =============================================================================

final_path = os.path.join(OUTPUT_DIR, "final_adapter")
trainer.model.save_pretrained(final_path)
tokenizer.save_pretrained(final_path)

# Save prompt template
with open(os.path.join(final_path, "prompt_config.json"), 'w') as f:
    json.dump({"prompt_template": PROMPT_TEMPLATE}, f)

logger.info(f"\nSaved to: {final_path}")

# =============================================================================
# TEST
# =============================================================================

logger.info("\n" + "="*80)
logger.info("TESTING FINE-TUNED MODEL")
logger.info("="*80)

model.eval()

test_captions = [
    "a woman with purple hair",
    "a horse pulling a trolley",
    "two people sitting on a bench",
    "a hotel room with a bed, chair, and a window",
]

for caption in test_captions:
    prompt = PROMPT_TEMPLATE.format(original=caption)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Corrected:" in generated:
        response = generated.split("### Corrected:")[-1].strip()
    else:
        response = generated[len(prompt):].strip()
    
    response = response.split("\n")[0].strip()
    
    logger.info(f"\n  Input:  '{caption}'")
    logger.info(f"  Output: '{response}'")
    
    # Check for garbage
    if "![" in response or "!@" in response or "http" in response:
        logger.warning("  ^ GARBAGE DETECTED")

logger.info("\n" + "="*80)
logger.info("COMPLETE")
logger.info("="*80)