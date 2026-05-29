"""
Universal Validation Script
---------------------------
Works with both Qwen and Mistral adapters.
Reads prompt_config.json to use correct format.
"""

import json
import torch
import logging
import sys
import os
from datetime import datetime
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

# =============================================================================
# ARGS
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", type=str, required=True, help="Path to adapter")
parser.add_argument("--data_file", type=str, default="data/output/results.json")
parser.add_argument("--num_examples", type=int, default=10)
args = parser.parse_args()

# =============================================================================
# LOGGING
# =============================================================================

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# LOAD CONFIG
# =============================================================================

BASE_DIR = os.getcwd()
ADAPTER_PATH = os.path.join(BASE_DIR, args.adapter_path)

if not os.path.exists(ADAPTER_PATH):
    logger.error(f"Adapter not found: {ADAPTER_PATH}")
    exit(1)

# Read prompt config
config_path = os.path.join(ADAPTER_PATH, "prompt_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    PROMPT_TEMPLATE = config["prompt_template"]
    MODEL_ID = config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
    logger.info(f"Loaded config from: {config_path}")
else:
    # Default to Qwen format
    PROMPT_TEMPLATE = "### Input:\n{original}\n\n### Corrected:\n"
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
    logger.warning("No config found, using defaults")

logger.info("="*80)
logger.info("        VALIDATION")
logger.info("="*80)
logger.info(f"Model: {MODEL_ID}")
logger.info(f"Adapter: {ADAPTER_PATH}")
logger.info(f"Template: {PROMPT_TEMPLATE[:50]}...")
logger.info("="*80)

# =============================================================================
# LOAD MODEL
# =============================================================================

logger.info(f"\nLoading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

logger.info("Model loaded.")

# =============================================================================
# GENERATION
# =============================================================================

def generate_correction(caption):
    prompt = PROMPT_TEMPLATE.format(original=caption)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.15,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response based on template
    if "### Corrected:" in generated:
        response = generated.split("### Corrected:")[-1].strip()
    elif "[/INST]" in generated:
        response = generated.split("[/INST]")[-1].strip()
    elif "Corrected:" in generated:
        response = generated.split("Corrected:")[-1].strip()
    else:
        response = generated[len(prompt):].strip()
    
    # Clean
    response = response.split("\n")[0].strip()
    response = response.split("###")[0].strip()  # Remove any next section
    
    return response

def is_garbage(text):
    """Check if output is garbage"""
    garbage_patterns = ["![", "!@", "!%", "http", "www.", "!!!!", ".jpg!", ".png!"]
    return any(p in text.lower() for p in garbage_patterns)

# =============================================================================
# LOAD DATA
# =============================================================================

DATA_FILE = os.path.join(BASE_DIR, args.data_file)

with open(DATA_FILE) as f:
    data = json.load(f)

examples = [d for d in data if any(v['hallucination_type'] != "None" for v in d['verification'])]
examples = examples[:args.num_examples]

logger.info(f"\nTesting on {len(examples)} examples")

# =============================================================================
# VALIDATE
# =============================================================================

logger.info("\n" + "="*80)
logger.info("RESULTS")
logger.info("="*80)

results = []
garbage_count = 0

for idx, ex in enumerate(examples, 1):
    image_id = ex['image_id']
    original = ex['caption']
    halls = [v['claim'] for v in ex['verification'] if v['hallucination_type'] != "None"]
    
    correction = generate_correction(original)
    
    is_bad = is_garbage(correction)
    if is_bad:
        garbage_count += 1
    
    # Check corrections
    removed = [h for h in halls if h.lower() not in correction.lower()]
    still_there = [h for h in halls if h.lower() in correction.lower()]
    rate = len(removed) / len(halls) * 100 if halls else 0
    
    logger.info(f"\n[{idx}/{len(examples)}] {image_id}")
    logger.info(f"  Original:    {original}")
    logger.info(f"  Correction:  {correction}")
    if is_bad:
        logger.warning(f"  STATUS:      GARBAGE OUTPUT")
    else:
        logger.info(f"  Removed:     {removed}")
        logger.info(f"  Rate:        {rate:.0f}%")
    
    results.append({
        "image_id": image_id,
        "original": original,
        "correction": correction,
        "is_garbage": is_bad,
        "rate": rate if not is_bad else 0
    })

# =============================================================================
# SUMMARY
# =============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY")
logger.info("="*80)

clean_results = [r for r in results if not r['is_garbage']]
avg_rate = sum(r['rate'] for r in clean_results) / len(clean_results) if clean_results else 0

logger.info(f"\nTotal:         {len(results)}")
logger.info(f"Garbage:       {garbage_count}")
logger.info(f"Clean:         {len(clean_results)}")
logger.info(f"Avg Rate:      {avg_rate:.1f}%")

if garbage_count == 0:
    logger.info("\n SUCCESS - No garbage outputs!")
elif garbage_count < len(results) / 2:
    logger.warning(f"\n PARTIAL - {garbage_count} garbage outputs")
else:
    logger.error(f"\n FAILED - {garbage_count}/{len(results)} garbage outputs")

# Save
output_file = os.path.join(BASE_DIR, "data/output/validation_results.json")
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
logger.info(f"\nSaved: {output_file}")