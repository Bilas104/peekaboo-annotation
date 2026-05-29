import json
import torch
import logging
import sys
import os
from datetime import datetime
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse

# --- LOGGING SETUP ---
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

# --- CONFIG ---
parser = argparse.ArgumentParser(description="Validate fine-tuned Qwen model on hallucination correction")
parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                    help="Base model ID")
parser.add_argument("--adapter_path", type=str, default="checkpoints_qwen/final_adapter",
                    help="Path to LoRA adapter")
parser.add_argument("--data_file", type=str, default="data/output/results.json",
                    help="AI results to validate against")
parser.add_argument("--num_examples", type=int, default=10,
                    help="Number of examples to validate")
parser.add_argument("--compare_baseline", action="store_true",
                    help="Also run baseline model without fine-tuning")
parser.add_argument("--save_corrections", type=str, default="data/output/model_corrections.json",
                    help="Save corrections to file")
args = parser.parse_args()

# --- HEADER ---
logger.info("="*80)
logger.info("              QWEN MODEL VALIDATION REPORT")
logger.info("="*80)
logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Base Model: {args.model_id}")
logger.info(f"Adapter: {args.adapter_path}")
logger.info(f"Data File: {args.data_file}")
logger.info("="*80)

# --- CHECK PATHS ---
BASE_DIR = os.getcwd()
ADAPTER_PATH = os.path.join(BASE_DIR, args.adapter_path)
DATA_FILE = os.path.join(BASE_DIR, args.data_file)

if not os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):
    logger.error(f"❌ Adapter not found at {ADAPTER_PATH}")
    logger.info("Please train the model first: python train_qwen.py")
    exit(1)

if not os.path.exists(DATA_FILE):
    logger.error(f"❌ Data file not found: {DATA_FILE}")
    exit(1)

logger.info("\n✅ All required files found")

# --- LOAD MODEL ---
logger.info("\n" + "="*80)
logger.info("                    LOADING MODELS")
logger.info("="*80)

logger.info(f"\n🔧 Loading base model: {args.model_id}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(args.model_id)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    args.model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager"
)

logger.info(f"✅ Base model loaded")
logger.info(f"\n🎯 Loading fine-tuned adapter from: {ADAPTER_PATH}...")

finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
finetuned_model.eval()

logger.info(f"✅ Fine-tuned model loaded successfully")

# Optional: Load baseline for comparison
baseline_model = None
if args.compare_baseline:
    logger.info(f"\n📊 Loading baseline model for comparison...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager"
    )
    baseline_model.eval()
    logger.info(f"✅ Baseline model loaded")

# --- LOAD DATA ---
logger.info(f"\n📂 Loading validation data from: {DATA_FILE}")
with open(DATA_FILE) as f:
    data = json.load(f)

# Filter examples with hallucinations
examples_with_halls = [
    d for d in data 
    if any(v['hallucination_type'] != "None" and v.get('confidence', 0) > 0.5 
           for v in d['verification'])
]

if not examples_with_halls:
    logger.error("❌ No examples with hallucinations found in dataset")
    exit(1)

logger.info(f"✅ Found {len(examples_with_halls)} images with hallucinations")
logger.info(f"📊 Validating on {min(args.num_examples, len(examples_with_halls))} examples")

# Select diverse examples
validation_examples = examples_with_halls[:args.num_examples]

# --- VALIDATION FUNCTION ---
def generate_correction(model, caption, tokenizer, max_new_tokens=100):
    """Generate corrected caption from model"""
    # messages = [
    #     {"role": "user", "content": f"Correct this caption: {caption}"}
    # ]
    messages = [
    {"role": "user", "content": f"Fix the hallucinations in this image caption:\n{caption}"}
]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's reply
    if "assistant\n" in out_text:
        correction = out_text.split("assistant\n")[-1].strip()
    else:
        correction = out_text.replace(text, "").strip()
    
    return correction

# --- RUN VALIDATION ---
logger.info("\n" + "="*80)
logger.info("                  VALIDATION RESULTS")
logger.info("="*80)

corrections_data = []

for idx, example in enumerate(validation_examples, 1):
    logger.info(f"\n{'='*80}")
    logger.info(f"EXAMPLE {idx}/{len(validation_examples)}")
    logger.info(f"{'='*80}")
    
    image_id = example['image_id']
    original_caption = example['caption']
    
    # Get hallucination details
    hallucinations = [
        v for v in example['verification'] 
        if v['hallucination_type'] != "None" and v.get('confidence', 0) > 0.5
    ]
    
    logger.info(f"\nImage: {image_id}")
    logger.info(f"\nOriginal Caption:")
    logger.info(f"   \"{original_caption}\"")
    
    logger.info(f"\nDetected Hallucinations ({len(hallucinations)}):")
    for h in hallucinations:
        logger.info(f"   • {h['claim']} - Type: {h['hallucination_type']} (Conf: {h.get('confidence', 0):.2f})")
        if h.get('expected_color'):
            logger.info(f"     Expected: {h['expected_color']}, Detected: {h.get('detected_colors', 'N/A')}")
        if h.get('expected_count') and h['expected_count'] > 1:
            logger.info(f"     Count: Expected {h['expected_count']}, Detected {h['detected_count']}")
    
    # Generate correction with fine-tuned model
    logger.info(f"\nFine-tuned Model Correction:")
    finetuned_correction = generate_correction(finetuned_model, original_caption, tokenizer)
    logger.info(f"   \"{finetuned_correction}\"")
    
    # Baseline comparison (if enabled)
    baseline_correction = None
    if baseline_model:
        logger.info(f"\nBaseline Model Correction:")
        baseline_correction = generate_correction(baseline_model, original_caption, tokenizer)
        logger.info(f"   \"{baseline_correction}\"")
    
    # Analyze correction quality
    logger.info(f"\nAnalysis:")
    
    # Check if hallucinated terms were removed/modified
    corrections_made = []
    still_present = []
    
    for h in hallucinations:
        claim = h['claim'].lower()
        if claim not in finetuned_correction.lower():
            corrections_made.append(claim)
        else:
            still_present.append(claim)
    
    if corrections_made:
        logger.info(f"   Removed/Modified: {', '.join(corrections_made)}")
    if still_present:
        logger.info(f"   Still Present: {', '.join(still_present)}")
    
    correction_rate = len(corrections_made) / len(hallucinations) * 100 if hallucinations else 0
    logger.info(f"   Correction Rate: {correction_rate:.1f}%")
    
    # Store results
    corrections_data.append({
        "image_id": image_id,
        "original_caption": original_caption,
        "finetuned_correction": finetuned_correction,
        "baseline_correction": baseline_correction,
        "hallucinations": [
            {
                "claim": h['claim'],
                "type": h['hallucination_type'],
                "confidence": h.get('confidence', 0)
            } for h in hallucinations
        ],
        "corrections_made": corrections_made,
        "still_present": still_present,
        "correction_rate": correction_rate
    })

# --- SUMMARY STATISTICS ---
logger.info("\n" + "="*80)
logger.info("                    SUMMARY STATISTICS")
logger.info("="*80)

total_hallucinations = sum(len(c['hallucinations']) for c in corrections_data)
total_corrections = sum(len(c['corrections_made']) for c in corrections_data)
avg_correction_rate = sum(c['correction_rate'] for c in corrections_data) / len(corrections_data) if corrections_data else 0

logger.info(f"\nOverall Performance:")
logger.info(f"   Total Examples: {len(corrections_data)}")
logger.info(f"   Total Hallucinations: {total_hallucinations}")
logger.info(f"   Total Corrections Made: {total_corrections}")
logger.info(f"   Average Correction Rate: {avg_correction_rate:.1f}%")

# Hallucination type breakdown
type_stats = {}
for c in corrections_data:
    for h in c['hallucinations']:
        h_type = h['type']
        if h_type not in type_stats:
            type_stats[h_type] = {"total": 0, "corrected": 0}
        type_stats[h_type]["total"] += 1
        if h['claim'].lower() in [cm.lower() for cm in c['corrections_made']]:
            type_stats[h_type]["corrected"] += 1

logger.info(f"\nPerformance by Hallucination Type:")
for h_type, stats in sorted(type_stats.items()):
    rate = stats['corrected'] / stats['total'] * 100 if stats['total'] > 0 else 0
    logger.info(f"   {h_type:25s}: {stats['corrected']}/{stats['total']:2d} corrected ({rate:5.1f}%)")

# Confidence analysis
high_conf_halls = [h for c in corrections_data for h in c['hallucinations'] if h['confidence'] > 0.8]
high_conf_corrected = sum(1 for c in corrections_data for h in c['hallucinations'] 
                          if h['confidence'] > 0.8 and h['claim'].lower() in [cm.lower() for cm in c['corrections_made']])

if high_conf_halls:
    logger.info(f"\nHigh Confidence Hallucinations (>0.8):")
    logger.info(f"   Total: {len(high_conf_halls)}")
    logger.info(f"   Corrected: {high_conf_corrected} ({high_conf_corrected/len(high_conf_halls)*100:.1f}%)")

# --- SAVE CORRECTIONS ---
if args.save_corrections:
    save_path = os.path.join(BASE_DIR, args.save_corrections)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(corrections_data, f, indent=2)
    
    logger.info(f"\nCorrections saved to: {save_path}")

# --- RECOMMENDATIONS ---
logger.info("\n" + "="*80)
logger.info("                      RECOMMENDATIONS")
logger.info("="*80)

recommendations = []

if avg_correction_rate < 50:
    recommendations.append("Low correction rate (<50%). Model may need more training data or longer training.")
elif avg_correction_rate < 70:
    recommendations.append("Moderate correction rate. Consider collecting more diverse training examples.")
else:
    recommendations.append("Good correction rate (>70%). Model is performing well.")

if still_present_count := sum(len(c['still_present']) for c in corrections_data):
    recommendations.append(f"{still_present_count} hallucinations still present. Review difficult cases.")

if high_conf_halls and high_conf_corrected / len(high_conf_halls) < 0.8:
    recommendations.append("Poor performance on high-confidence hallucinations. Verify detection pipeline.")

# Check for overcorrection (caption became too short)
avg_original_len = sum(len(c['original_caption'].split()) for c in corrections_data) / len(corrections_data)
avg_corrected_len = sum(len(c['finetuned_correction'].split()) for c in corrections_data) / len(corrections_data)

if avg_corrected_len < avg_original_len * 0.6:
    recommendations.append("Model may be over-correcting (captions too short). Review training approach.")

if not recommendations:
    recommendations.append("Model performance looks good across all metrics.")

logger.info("")
for rec in recommendations:
    logger.info(f"   {rec}")

# --- FOOTER ---
logger.info("\n" + "="*80)
logger.info(f"Full validation report saved to: {log_filename}")
logger.info(f"Validation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

logger.info("\nNext Steps:")
logger.info("   1. Review examples with low correction rates")
logger.info("   2. Collect human annotations for ground truth comparison")
logger.info("   3. Use improved_calc_metrics.py --human_file to compare with human labels")
logger.info("   4. Consider DPO training with preference pairs if needed")