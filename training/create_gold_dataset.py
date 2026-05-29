"""
Script to create gold training dataset from human annotations.
This converts human-annotated data into a format suitable for fine-tuning.
"""

import json
import argparse
import logging
import os
import sys
from datetime import datetime
from collections import defaultdict

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/gold_dataset_creation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Create gold training dataset from human annotations")
parser.add_argument("--human_file", type=str, default="data/output/human_annotations.json",
                    help="Path to human annotations")
parser.add_argument("--ai_file", type=str, default="data/output/results.json",
                    help="Path to AI results (for reference)")
parser.add_argument("--output_file", type=str, default="data/output/human_gold_dataset.json",
                    help="Output path for gold dataset")
parser.add_argument("--min_corrections", type=int, default=1,
                    help="Minimum number of corrections needed to include example")
parser.add_argument("--strategy", type=str, default="remove",
                    choices=["remove", "replace", "rephrase"],
                    help="Correction strategy: remove, replace with generic, or rephrase")
args = parser.parse_args()

# --- HEADER ---
logger.info("="*80)
logger.info("           GOLD DATASET CREATION FOR FINE-TUNING")
logger.info("="*80)
logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Human Annotations: {args.human_file}")
logger.info(f"Strategy: {args.strategy}")
logger.info("="*80)

# --- LOAD DATA ---
if not os.path.exists(args.human_file):
    logger.error(f"❌ Human annotation file not found: {args.human_file}")
    exit(1)

if not os.path.exists(args.ai_file):
    logger.error(f"❌ AI results file not found: {args.ai_file}")
    exit(1)

with open(args.human_file) as f:
    human_data = json.load(f)

with open(args.ai_file) as f:
    ai_data = json.load(f)

# Create lookup dictionary
ai_dict = {item['image_id']: item for item in ai_data}

logger.info(f"\n✅ Loaded {len(human_data)} human annotations")
logger.info(f"✅ Loaded {len(ai_data)} AI results")

# --- CORRECTION STRATEGIES ---
def remove_hallucinations(caption, hallucinated_claims):
    """Remove hallucinated phrases from caption"""
    corrected = caption
    for claim in sorted(hallucinated_claims, key=len, reverse=True):  # Remove longer phrases first
        # Try exact match first
        if claim in corrected:
            corrected = corrected.replace(claim, "").strip()
        else:
            # Try case-insensitive
            import re
            pattern = re.compile(re.escape(claim), re.IGNORECASE)
            corrected = pattern.sub("", corrected)
    
    # Clean up extra spaces and punctuation
    corrected = " ".join(corrected.split())
    corrected = corrected.replace(" ,", ",").replace(" .", ".")
    corrected = corrected.replace("  ", " ").strip()
    
    return corrected

def replace_with_generic(caption, hallucinated_claims):
    """Replace hallucinated phrases with generic terms"""
    corrected = caption
    generic_replacements = {
        # These are learned patterns - could be expanded
        "red": "colored",
        "blue": "colored",
        "green": "colored",
        "yellow": "colored",
        "two": "some",
        "three": "several",
        "four": "several",
        "five": "many"
    }
    
    for claim in sorted(hallucinated_claims, key=len, reverse=True):
        # Check if it's a color or count issue
        for specific, generic in generic_replacements.items():
            if specific in claim.lower():
                corrected = corrected.replace(claim, claim.replace(specific, generic))
                break
        else:
            # If no replacement found, remove
            corrected = corrected.replace(claim, "")
    
    # Clean up
    corrected = " ".join(corrected.split())
    corrected = corrected.replace(" ,", ",").replace(" .", ".")
    corrected = corrected.replace("  ", " ").strip()
    
    return corrected

def rephrase_caption(caption, hallucinated_claims, verification_details):
    """
    Rephrase caption by keeping accurate parts and rephrasing hallucinated parts.
    This is more sophisticated but requires more logic.
    """
    # For now, use replace strategy as base
    corrected = replace_with_generic(caption, hallucinated_claims)
    
    # Additional rephrasing logic could be added here
    # For example, restructuring sentences, combining phrases, etc.
    
    return corrected

# --- PROCESS ANNOTATIONS ---
logger.info("\n" + "="*80)
logger.info("                PROCESSING ANNOTATIONS")
logger.info("="*80)

gold_dataset = []
stats = {
    "total_processed": 0,
    "with_hallucinations": 0,
    "corrections_made": 0,
    "skipped": 0,
    "hallucination_types": defaultdict(int)
}

for human_item in human_data:
    stats["total_processed"] += 1
    image_id = human_item['image_id']
    original_caption = human_item['caption']
    human_labels = human_item['human_labels']
    
    # Find hallucinations according to human
    human_hallucinations = [
        label for label in human_labels 
        if label['label'] == "Hallucination"
    ]
    
    if not human_hallucinations:
        stats["skipped"] += 1
        continue
    
    if len(human_hallucinations) < args.min_corrections:
        stats["skipped"] += 1
        continue
    
    stats["with_hallucinations"] += 1
    
    # Extract hallucinated claims
    hallucinated_claims = [h['claim'] for h in human_hallucinations]
    
    # Get additional context from AI results
    verification_details = None
    if image_id in ai_dict:
        verification_details = ai_dict[image_id]['verification']
    
    # Apply correction strategy
    if args.strategy == "remove":
        corrected_caption = remove_hallucinations(original_caption, hallucinated_claims)
    elif args.strategy == "replace":
        corrected_caption = replace_with_generic(original_caption, hallucinated_claims)
    elif args.strategy == "rephrase":
        corrected_caption = rephrase_caption(original_caption, hallucinated_claims, verification_details)
    else:
        corrected_caption = original_caption
    
    # Ensure correction actually changed something
    if corrected_caption == original_caption:
        logger.warning(f"⚠️  No change for {image_id}: '{original_caption}'")
        stats["skipped"] += 1
        continue
    
    # Ensure corrected caption is not too short
    if len(corrected_caption.split()) < 3:
        logger.warning(f"⚠️  Corrected caption too short for {image_id}: '{corrected_caption}'")
        stats["skipped"] += 1
        continue
    
    stats["corrections_made"] += 1
    
    # Track hallucination types
    for h in human_hallucinations:
        h_type = h.get('ai_prediction', 'Unknown')
        stats["hallucination_types"][h_type] += 1
    
    # Add to gold dataset
    gold_dataset.append({
        "image_id": image_id,
        "original_caption": original_caption,
        "human_corrected_caption": corrected_caption,
        "hallucinations_corrected": hallucinated_claims,
        "hallucination_types": [h.get('ai_prediction', 'Unknown') for h in human_hallucinations],
        "correction_count": len(human_hallucinations),
        "notes": human_item.get('notes', ''),
        "timestamp": human_item.get('timestamp', '')
    })
    
    # Log progress every 10 items
    if len(gold_dataset) % 10 == 0:
        logger.info(f"   Processed {stats['total_processed']} annotations, created {len(gold_dataset)} training examples...")

# --- STATISTICS ---
logger.info("\n" + "="*80)
logger.info("                    STATISTICS")
logger.info("="*80)

logger.info(f"\n📊 Processing Summary:")
logger.info(f"   Total Annotations Processed:      {stats['total_processed']}")
logger.info(f"   Annotations with Hallucinations:  {stats['with_hallucinations']}")
logger.info(f"   Training Examples Created:        {stats['corrections_made']}")
logger.info(f"   Skipped (no change/too short):    {stats['skipped']}")

if stats['corrections_made'] > 0:
    avg_corrections_per_example = sum(item['correction_count'] for item in gold_dataset) / len(gold_dataset)
    logger.info(f"   Avg Corrections per Example:      {avg_corrections_per_example:.2f}")

logger.info(f"\n📈 Hallucination Types in Training Data:")
for h_type, count in sorted(stats['hallucination_types'].items(), key=lambda x: x[1], reverse=True):
    percentage = count / sum(stats['hallucination_types'].values()) * 100
    logger.info(f"   {h_type:25s}: {count:4d} ({percentage:5.1f}%)")

# --- SAMPLE EXAMPLES ---
if gold_dataset:
    logger.info("\n" + "="*80)
    logger.info("                 SAMPLE EXAMPLES")
    logger.info("="*80)
    
    # Show first 5 examples
    for i, example in enumerate(gold_dataset[:5], 1):
        logger.info(f"\nExample {i}:")
        logger.info(f"   Image: {example['image_id']}")
        logger.info(f"   Original:  '{example['original_caption']}'")
        logger.info(f"   Corrected: '{example['human_corrected_caption']}'")
        logger.info(f"   Hallucinations: {', '.join(example['hallucinations_corrected'])}")
        logger.info(f"   Types: {', '.join(set(example['hallucination_types']))}")

# --- SAVE DATASET ---
if gold_dataset:
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w') as f:
        json.dump(gold_dataset, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ Gold dataset saved to: {args.output_file}")
    logger.info(f"   Total training examples: {len(gold_dataset)}")
    logger.info("="*80)
else:
    logger.error("\n❌ No training examples created!")
    logger.info("   Possible reasons:")
    logger.info("   - All annotations had no hallucinations")
    logger.info("   - Corrections resulted in identical or too-short captions")
    logger.info("   - min_corrections threshold too high")

# --- QUALITY CHECKS ---
logger.info("\n" + "="*80)
logger.info("                  QUALITY CHECKS")
logger.info("="*80)

if gold_dataset:
    # Check caption length distribution
    original_lengths = [len(item['original_caption'].split()) for item in gold_dataset]
    corrected_lengths = [len(item['human_corrected_caption'].split()) for item in gold_dataset]
    
    logger.info(f"\n📏 Caption Length Analysis:")
    logger.info(f"   Original  - Mean: {sum(original_lengths)/len(original_lengths):.1f}, "
                f"Min: {min(original_lengths)}, Max: {max(original_lengths)}")
    logger.info(f"   Corrected - Mean: {sum(corrected_lengths)/len(corrected_lengths):.1f}, "
                f"Min: {min(corrected_lengths)}, Max: {max(corrected_lengths)}")
    
    # Check for potential issues
    issues = []
    
    # Issue 1: Too many short corrected captions
    short_captions = sum(1 for length in corrected_lengths if length < 5)
    if short_captions > len(gold_dataset) * 0.3:
        issues.append(f"⚠️  {short_captions} corrected captions are very short (<5 words)")
    
    # Issue 2: Identical captions
    identical_count = sum(1 for item in gold_dataset 
                         if item['original_caption'] == item['human_corrected_caption'])
    if identical_count > 0:
        issues.append(f"⚠️  {identical_count} examples have identical original and corrected captions")
    
    # Issue 3: Single correction examples dominate
    single_correction_count = sum(1 for item in gold_dataset if item['correction_count'] == 1)
    if single_correction_count > len(gold_dataset) * 0.8:
        issues.append(f"⚠️  {single_correction_count} examples have only 1 correction (limited diversity)")
    
    if issues:
        logger.info(f"\n⚠️  Potential Quality Issues:")
        for issue in issues:
            logger.info(f"   {issue}")
    else:
        logger.info(f"\n✅ No major quality issues detected")

# --- RECOMMENDATIONS ---
logger.info("\n" + "="*80)
logger.info("                   RECOMMENDATIONS")
logger.info("="*80)

recommendations = []

if not gold_dataset:
    recommendations.append("❌ No training data created. Collect more human annotations first.")
elif len(gold_dataset) < 50:
    recommendations.append("⚠️  Very small training set (<50). Collect more annotations for better results.")
    recommendations.append("   Consider lowering --min_corrections threshold.")
elif len(gold_dataset) < 100:
    recommendations.append("⚠️  Small training set (50-100). More data recommended for robust fine-tuning.")
else:
    recommendations.append(f"✅ Good training set size ({len(gold_dataset)} examples).")

if stats['corrections_made'] > 0:
    if args.strategy == "remove":
        recommendations.append("   Using 'remove' strategy. Consider 'replace' for more natural captions.")
    elif args.strategy == "replace":
        recommendations.append("✅ Using 'replace' strategy - good balance of correction and naturalness.")

if gold_dataset and len(set(item['correction_count'] for item in gold_dataset)) == 1:
    recommendations.append("⚠️  All examples have same correction count. Seek more diverse cases.")

logger.info("")
for rec in recommendations:
    logger.info(f"   {rec}")

# --- NEXT STEPS ---
logger.info("\n" + "="*80)
logger.info("                     NEXT STEPS")
logger.info("="*80)

if gold_dataset:
    logger.info("\n✅ Ready for fine-tuning! Run:")
    logger.info(f"   python train_qwen.py")
    logger.info("\n   This will train a hallucination correction model using your gold dataset.")
else:
    logger.info("\n❌ Dataset creation failed. Please:")
    logger.info("   1. Collect more human annotations using: streamlit run improved_game_ui.py")
    logger.info("   2. Ensure annotations include hallucination labels")
    logger.info("   3. Re-run this script")

# --- FOOTER ---
logger.info("\n" + "="*80)
logger.info(f"📄 Log saved to: {log_filename}")
logger.info(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)