"""
Validation on Human-Verified Hallucinations
--------------------------------------------
Only tests on images where:
1. Human marked something as hallucination
2. AI also detected it as hallucination
3. Human provided a corrected caption

This gives us ground truth to compare against.
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
parser.add_argument("--adapter_path", type=str, required=True)
parser.add_argument("--human_gold_file", type=str, default="data/output/human_gold_dataset.json")
parser.add_argument("--ai_results_file", type=str, default="data/output/results.json")
parser.add_argument("--num_examples", type=int, default=20)
args = parser.parse_args()

# =============================================================================
# LOGGING
# =============================================================================

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/validation_human_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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

config_path = os.path.join(ADAPTER_PATH, "prompt_config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    PROMPT_TEMPLATE = config["prompt_template"]
    MODEL_ID = config.get("model_id", "Qwen/Qwen2.5-7B-Instruct")
else:
    PROMPT_TEMPLATE = "### Input:\n{original}\n\n### Corrected:\n"
    MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

logger.info("="*80)
logger.info("   VALIDATION ON HUMAN-VERIFIED HALLUCINATIONS")
logger.info("="*80)

# =============================================================================
# LOAD HUMAN GOLD DATA
# =============================================================================

HUMAN_FILE = os.path.join(BASE_DIR, args.human_gold_file)
AI_FILE = os.path.join(BASE_DIR, args.ai_results_file)

if not os.path.exists(HUMAN_FILE):
    logger.error(f"Human gold file not found: {HUMAN_FILE}")
    exit(1)

with open(HUMAN_FILE) as f:
    human_gold = json.load(f)

logger.info(f"Loaded {len(human_gold)} human-annotated examples")

# Load AI results for cross-reference
ai_dict = {}
if os.path.exists(AI_FILE):
    with open(AI_FILE) as f:
        ai_data = json.load(f)
    ai_dict = {item['image_id']: item for item in ai_data}
    logger.info(f"Loaded {len(ai_dict)} AI results for cross-reference")

# =============================================================================
# FIND AGREED HALLUCINATIONS
# =============================================================================

agreed_examples = []

for item in human_gold:
    image_id = item['image_id']
    original = item['original_caption']
    human_corrected = item['human_corrected_caption']
    human_hallucinations = item['hallucinations_corrected']
    
    # Check if AI also flagged this image
    if image_id in ai_dict:
        ai_item = ai_dict[image_id]
        ai_hallucinations = [
            v['claim'] for v in ai_item['verification']
            if v['hallucination_type'] != "None"
        ]
        
        # Find overlap - claims both human and AI flagged
        agreed_claims = []
        for h_claim in human_hallucinations:
            for a_claim in ai_hallucinations:
                # Check if claims match (fuzzy)
                if h_claim.lower() in a_claim.lower() or a_claim.lower() in h_claim.lower():
                    agreed_claims.append(h_claim)
                    break
        
        if agreed_claims:
            agreed_examples.append({
                "image_id": image_id,
                "original": original,
                "human_corrected": human_corrected,
                "human_hallucinations": human_hallucinations,
                "ai_hallucinations": ai_hallucinations,
                "agreed_hallucinations": agreed_claims
            })

logger.info(f"\nFound {len(agreed_examples)} images with AGREED hallucinations (human + AI)")

if not agreed_examples:
    logger.error("No agreed examples found! Check data files.")
    exit(1)

# Limit to requested number
test_examples = agreed_examples[:args.num_examples]
logger.info(f"Testing on {len(test_examples)} examples")

# =============================================================================
# LOAD MODEL
# =============================================================================

logger.info(f"\nLoading model: {MODEL_ID}")

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
    
    if "### Corrected:" in generated:
        response = generated.split("### Corrected:")[-1].strip()
    elif "[/INST]" in generated:
        response = generated.split("[/INST]")[-1].strip()
    else:
        response = generated[len(prompt):].strip()
    
    response = response.split("\n")[0].strip()
    response = response.split("###")[0].strip()
    
    return response

def is_garbage(text):
    garbage_patterns = ["![", "!@", "!%", "http", "www.", "!!!!"]
    return any(p in text.lower() for p in garbage_patterns)

def similarity_score(text1, text2):
    """Simple word overlap similarity"""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    if not words1 or not words2:
        return 0.0
    intersection = words1 & words2
    union = words1 | words2
    return len(intersection) / len(union)

# =============================================================================
# VALIDATE
# =============================================================================

logger.info("\n" + "="*80)
logger.info("RESULTS")
logger.info("="*80)

results = []

for idx, ex in enumerate(test_examples, 1):
    image_id = ex['image_id']
    original = ex['original']
    human_corrected = ex['human_corrected']
    agreed_halls = ex['agreed_hallucinations']
    
    # Generate model correction
    model_correction = generate_correction(original)
    
    # Check quality
    is_bad = is_garbage(model_correction)
    
    # Compare to human correction
    sim_to_human = similarity_score(model_correction, human_corrected)
    sim_to_original = similarity_score(model_correction, original)
    
    # Check if hallucinations were addressed
    halls_removed = []
    halls_still_present = []
    for hall in agreed_halls:
        if hall.lower() not in model_correction.lower():
            halls_removed.append(hall)
        else:
            halls_still_present.append(hall)
    
    removal_rate = len(halls_removed) / len(agreed_halls) * 100 if agreed_halls else 0
    
    logger.info(f"\n{'='*60}")
    logger.info(f"[{idx}/{len(test_examples)}] {image_id}")
    logger.info(f"{'='*60}")
    logger.info(f"  Original:        \"{original}\"")
    logger.info(f"  Human Corrected: \"{human_corrected}\"")
    logger.info(f"  Model Output:    \"{model_correction}\"")
    logger.info(f"  ")
    logger.info(f"  Agreed Hallucinations: {agreed_halls}")
    logger.info(f"  Removed by Model:      {halls_removed}")
    logger.info(f"  Still Present:         {halls_still_present}")
    logger.info(f"  ")
    logger.info(f"  Similarity to Human:   {sim_to_human:.2f}")
    logger.info(f"  Similarity to Original:{sim_to_original:.2f}")
    logger.info(f"  Hallucination Removal: {removal_rate:.0f}%")
    
    if is_bad:
        logger.warning(f"  STATUS: GARBAGE OUTPUT")
    elif sim_to_human > 0.6:
        logger.info(f"  STATUS: GOOD (close to human)")
    elif removal_rate > 50:
        logger.info(f"  STATUS: OK (hallucinations addressed)")
    else:
        logger.info(f"  STATUS: NEEDS REVIEW")
    
    results.append({
        "image_id": image_id,
        "original": original,
        "human_corrected": human_corrected,
        "model_correction": model_correction,
        "agreed_hallucinations": agreed_halls,
        "halls_removed": halls_removed,
        "halls_still_present": halls_still_present,
        "removal_rate": removal_rate,
        "sim_to_human": sim_to_human,
        "sim_to_original": sim_to_original,
        "is_garbage": is_bad
    })

# =============================================================================
# SUMMARY
# =============================================================================

logger.info("\n" + "="*80)
logger.info("SUMMARY STATISTICS")
logger.info("="*80)

clean_results = [r for r in results if not r['is_garbage']]
garbage_count = len(results) - len(clean_results)

if clean_results:
    avg_removal_rate = sum(r['removal_rate'] for r in clean_results) / len(clean_results)
    avg_sim_to_human = sum(r['sim_to_human'] for r in clean_results) / len(clean_results)
    avg_sim_to_original = sum(r['sim_to_original'] for r in clean_results) / len(clean_results)
    
    high_similarity = sum(1 for r in clean_results if r['sim_to_human'] > 0.5)
    full_removal = sum(1 for r in clean_results if r['removal_rate'] == 100)
    
    logger.info(f"\n  Total Examples:              {len(results)}")
    logger.info(f"  Garbage Outputs:             {garbage_count}")
    logger.info(f"  Clean Outputs:               {len(clean_results)}")
    logger.info(f"  ")
    logger.info(f"  Avg Hallucination Removal:   {avg_removal_rate:.1f}%")
    logger.info(f"  Avg Similarity to Human:     {avg_sim_to_human:.2f}")
    logger.info(f"  Avg Similarity to Original:  {avg_sim_to_original:.2f}")
    logger.info(f"  ")
    logger.info(f"  High Similarity (>0.5):      {high_similarity}/{len(clean_results)}")
    logger.info(f"  Full Hallucination Removal:  {full_removal}/{len(clean_results)}")
    
    # Overall assessment
    logger.info(f"\n" + "-"*40)
    if garbage_count == 0 and avg_removal_rate > 60 and avg_sim_to_human > 0.4:
        logger.info("  OVERALL: EXCELLENT - Model performs well!")
    elif garbage_count == 0 and (avg_removal_rate > 40 or avg_sim_to_human > 0.3):
        logger.info("  OVERALL: GOOD - Model works, some room for improvement")
    elif garbage_count == 0:
        logger.info("  OVERALL: FAIR - Clean outputs but low correction rate")
    else:
        logger.info("  OVERALL: NEEDS WORK - Garbage outputs present")
    logger.info("-"*40)

else:
    logger.error("All outputs were garbage!")

# =============================================================================
# SAVE RESULTS
# =============================================================================

output_file = os.path.join(BASE_DIR, "data/output/validation_human_agreed.json")
with open(output_file, 'w') as f:
    json.dump({
        "summary": {
            "total": len(results),
            "garbage": garbage_count,
            "clean": len(clean_results),
            "avg_removal_rate": avg_removal_rate if clean_results else 0,
            "avg_sim_to_human": avg_sim_to_human if clean_results else 0,
        },
        "results": results
    }, f, indent=2)

logger.info(f"\nResults saved to: {output_file}")
logger.info(f"Log saved to: {log_filename}")