"""
Comprehensive Peekaboo Metrics Report
--------------------------------------
Calculates:
1. CHAIR Metrics (AI Detection)
2. Human Agreement Metrics
3. Fine-tuned Model Correction Metrics
4. End-to-End Pipeline Metrics
"""

import json
import argparse
import logging
import os
import sys
from datetime import datetime
from collections import defaultdict
import numpy as np

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/comprehensive_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
parser = argparse.ArgumentParser(description="Calculate comprehensive hallucination metrics")
parser.add_argument("--ai_results", type=str, default="data/output/results.json",
                    help="Path to AI detection results")
parser.add_argument("--human_annotations", type=str, default="data/output/human_annotations.json",
                    help="Path to human annotations")
parser.add_argument("--human_gold", type=str, default="data/output/human_gold_dataset.json",
                    help="Path to human gold dataset (for training)")
parser.add_argument("--model_corrections", type=str, default="data/output/validation_human_agreed.json",
                    help="Path to model correction results")
parser.add_argument("--confidence_threshold", type=float, default=0.7,
                    help="Minimum confidence to count as hallucination")
args = parser.parse_args()

# --- HEADER ---
logger.info("="*80)
logger.info("          PEEKABOO: COMPREHENSIVE METRICS REPORT")
logger.info("="*80)
logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)

# =============================================================================
# SECTION 1: AI DETECTION METRICS (CHAIR)
# =============================================================================

logger.info("\n" + "="*80)
logger.info("  SECTION 1: AI HALLUCINATION DETECTION METRICS")
logger.info("="*80)

BASE_DIR = os.getcwd()
AI_FILE = os.path.join(BASE_DIR, args.ai_results)

if os.path.exists(AI_FILE):
    with open(AI_FILE) as f:
        ai_data = json.load(f)
    
    logger.info(f"\nDataset: {len(ai_data)} images processed")
    
    # CHAIR Metrics
    images_with_hallucination = 0
    total_hallucinated_objects = 0
    total_mentioned_objects = 0
    hallucination_types = defaultdict(int)
    confidence_scores = []
    
    # Detection stats
    detection_stats = {
        "total_claims": 0,
        "claims_with_detections": 0,
        "total_detections": 0,
        "detection_scores": []
    }
    
    # Color stats
    color_stats = {"total": 0, "matches": 0, "mismatches": 0}
    
    # Count stats
    count_stats = {"total": 0, "exact": 0, "within_1": 0, "mismatch": 0}
    
    for item in ai_data:
        has_hall = False
        
        for v in item['verification']:
            total_mentioned_objects += 1
            detection_stats["total_claims"] += 1
            
            h_type = v['hallucination_type']
            confidence = v.get('confidence', 0)
            detected_count = v['detected_count']
            expected_count = v.get('expected_count', 1)
            
            if detected_count > 0:
                detection_stats["claims_with_detections"] += 1
                detection_stats["total_detections"] += detected_count
                if 'detection_scores' in v and v['detection_scores']:
                    detection_stats["detection_scores"].extend(v['detection_scores'])
            
            # Color tracking
            if v.get('expected_color'):
                color_stats["total"] += 1
                if v.get('color_match') is True:
                    color_stats["matches"] += 1
                elif v.get('color_match') is False:
                    color_stats["mismatches"] += 1
            
            # Count tracking
            if expected_count > 1:
                count_stats["total"] += 1
                if detected_count == expected_count:
                    count_stats["exact"] += 1
                elif abs(detected_count - expected_count) <= 1:
                    count_stats["within_1"] += 1
                else:
                    count_stats["mismatch"] += 1
            
            # Hallucination detection
            if h_type != "None" and confidence >= args.confidence_threshold:
                total_hallucinated_objects += 1
                has_hall = True
                hallucination_types[h_type] += 1
                confidence_scores.append(confidence)
        
        if has_hall:
            images_with_hallucination += 1
    
    # Calculate CHAIR
    chair_i = images_with_hallucination / len(ai_data) if ai_data else 0
    chair_s = total_hallucinated_objects / total_mentioned_objects if total_mentioned_objects else 0
    
    logger.info(f"\n{'─'*60}")
    logger.info(f"  CHAIR METRICS (Lower is Better)")
    logger.info(f"{'─'*60}")
    logger.info(f"  CHAIR_i (Image-level):      {chair_i:6.2%}  ({images_with_hallucination}/{len(ai_data)} images)")
    logger.info(f"  CHAIR_s (Sentence-level):   {chair_s:6.2%}  ({total_hallucinated_objects}/{total_mentioned_objects} objects)")
    
    logger.info(f"\n{'─'*60}")
    logger.info(f"  DETECTION STATISTICS")
    logger.info(f"{'─'*60}")
    detection_coverage = detection_stats["claims_with_detections"] / detection_stats["total_claims"] if detection_stats["total_claims"] else 0
    avg_det_score = np.mean(detection_stats["detection_scores"]) if detection_stats["detection_scores"] else 0
    logger.info(f"  Total Claims:               {detection_stats['total_claims']}")
    logger.info(f"  Detection Coverage:         {detection_coverage:6.2%}")
    logger.info(f"  Avg Detection Score:        {avg_det_score:.3f}")
    
    logger.info(f"\n{'─'*60}")
    logger.info(f"  HALLUCINATION TYPE BREAKDOWN")
    logger.info(f"{'─'*60}")
    for h_type, count in sorted(hallucination_types.items(), key=lambda x: x[1], reverse=True):
        pct = count / total_hallucinated_objects * 100 if total_hallucinated_objects else 0
        logger.info(f"  {h_type:25s}  {count:4d}  ({pct:5.1f}%)")
    
    if color_stats["total"] > 0:
        color_acc = color_stats["matches"] / color_stats["total"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"  COLOR VERIFICATION")
        logger.info(f"{'─'*60}")
        logger.info(f"  Total Color Claims:         {color_stats['total']}")
        logger.info(f"  Accuracy:                   {color_acc:6.2%}")
    
    if count_stats["total"] > 0:
        count_acc = (count_stats["exact"] + count_stats["within_1"]) / count_stats["total"]
        logger.info(f"\n{'─'*60}")
        logger.info(f"  COUNT VERIFICATION")
        logger.info(f"{'─'*60}")
        logger.info(f"  Total Count Claims:         {count_stats['total']}")
        logger.info(f"  Exact Matches:              {count_stats['exact']}")
        logger.info(f"  Within ±1:                  {count_stats['within_1']}")
        logger.info(f"  Accuracy (with tolerance):  {count_acc:6.2%}")

else:
    logger.warning(f"AI results file not found: {AI_FILE}")
    ai_data = []

# =============================================================================
# SECTION 2: HUMAN ANNOTATION STATISTICS
# =============================================================================

logger.info("\n" + "="*80)
logger.info("  SECTION 2: HUMAN ANNOTATION STATISTICS")
logger.info("="*80)

HUMAN_FILE = os.path.join(BASE_DIR, args.human_annotations)
GOLD_FILE = os.path.join(BASE_DIR, args.human_gold)

if os.path.exists(HUMAN_FILE):
    with open(HUMAN_FILE) as f:
        human_data = json.load(f)
    
    total_annotations = len(human_data)
    total_claims_labeled = sum(len(item['human_labels']) for item in human_data)
    
    human_hall_count = 0
    human_accurate_count = 0
    human_hall_types = defaultdict(int)
    
    for item in human_data:
        for label in item['human_labels']:
            if label['label'] == "Hallucination":
                human_hall_count += 1
                h_type = label.get('ai_prediction', 'Unknown')
                human_hall_types[h_type] += 1
            else:
                human_accurate_count += 1
    
    logger.info(f"\n{'─'*60}")
    logger.info(f"  ANNOTATION SUMMARY")
    logger.info(f"{'─'*60}")
    logger.info(f"  Total Images Annotated:     {total_annotations}")
    logger.info(f"  Total Claims Labeled:       {total_claims_labeled}")
    logger.info(f"  Marked as Hallucination:    {human_hall_count}  ({human_hall_count/total_claims_labeled*100:.1f}%)")
    logger.info(f"  Marked as Accurate:         {human_accurate_count}  ({human_accurate_count/total_claims_labeled*100:.1f}%)")
    
    if human_hall_types:
        logger.info(f"\n{'─'*60}")
        logger.info(f"  HUMAN-IDENTIFIED HALLUCINATION TYPES")
        logger.info(f"{'─'*60}")
        for h_type, count in sorted(human_hall_types.items(), key=lambda x: x[1], reverse=True):
            pct = count / human_hall_count * 100 if human_hall_count else 0
            logger.info(f"  {h_type:25s}  {count:4d}  ({pct:5.1f}%)")

else:
    logger.warning(f"Human annotations file not found: {HUMAN_FILE}")
    human_data = []

# Gold dataset stats
if os.path.exists(GOLD_FILE):
    with open(GOLD_FILE) as f:
        gold_data = json.load(f)
    
    logger.info(f"\n{'─'*60}")
    logger.info(f"  GOLD TRAINING DATASET")
    logger.info(f"{'─'*60}")
    logger.info(f"  Training Examples:          {len(gold_data)}")
    
    avg_corrections = np.mean([item['correction_count'] for item in gold_data])
    logger.info(f"  Avg Corrections/Example:    {avg_corrections:.2f}")

# =============================================================================
# SECTION 3: AI vs HUMAN AGREEMENT
# =============================================================================

logger.info("\n" + "="*80)
logger.info("  SECTION 3: AI vs HUMAN AGREEMENT")
logger.info("="*80)

if ai_data and human_data:
    ai_dict = {item['image_id']: item for item in ai_data}
    
    y_true = []  # Human labels
    y_pred = []  # AI predictions
    
    agreement_details = {
        "true_positive": 0,   # Both say hallucination
        "true_negative": 0,   # Both say accurate
        "false_positive": 0,  # AI says hall, human says accurate
        "false_negative": 0   # AI says accurate, human says hall
    }
    
    for human_item in human_data:
        img_id = human_item['image_id']
        if img_id not in ai_dict:
            continue
        
        ai_item = ai_dict[img_id]
        human_labels = {h['claim']: h['label'] for h in human_item['human_labels']}
        
        for v in ai_item['verification']:
            claim = v['claim']
            if claim not in human_labels:
                continue
            
            ai_hall = v['hallucination_type'] != "None" and v.get('confidence', 0) >= args.confidence_threshold
            human_hall = human_labels[claim] == "Hallucination"
            
            y_true.append(1 if human_hall else 0)
            y_pred.append(1 if ai_hall else 0)
            
            if ai_hall and human_hall:
                agreement_details["true_positive"] += 1
            elif not ai_hall and not human_hall:
                agreement_details["true_negative"] += 1
            elif ai_hall and not human_hall:
                agreement_details["false_positive"] += 1
            else:
                agreement_details["false_negative"] += 1
    
    if y_true and y_pred:
        # Calculate metrics
        tp = agreement_details["true_positive"]
        tn = agreement_details["true_negative"]
        fp = agreement_details["false_positive"]
        fn = agreement_details["false_negative"]
        
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        
        # Cohen's Kappa
        p_o = accuracy
        p_e = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total) if total else 0
        kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else 0
        
        logger.info(f"\n{'─'*60}")
        logger.info(f"  CONFUSION MATRIX")
        logger.info(f"{'─'*60}")
        logger.info(f"                        Human: Hall    Human: Accurate")
        logger.info(f"  AI: Hallucination        {tp:4d}            {fp:4d}")
        logger.info(f"  AI: Accurate             {fn:4d}            {tn:4d}")
        
        logger.info(f"\n{'─'*60}")
        logger.info(f"  AGREEMENT METRICS")
        logger.info(f"{'─'*60}")
        logger.info(f"  Total Comparisons:          {total}")
        logger.info(f"  Accuracy:                   {accuracy:6.2%}")
        logger.info(f"  Precision:                  {precision:6.2%}")
        logger.info(f"  Recall:                     {recall:6.2%}")
        logger.info(f"  F1 Score:                   {f1:6.2%}")
        logger.info(f"  Cohen's Kappa:              {kappa:6.3f}  ({'Substantial' if kappa > 0.6 else 'Moderate' if kappa > 0.4 else 'Fair' if kappa > 0.2 else 'Slight'})")

else:
    logger.warning("Cannot calculate agreement - missing AI or human data")

# =============================================================================
# SECTION 4: FINE-TUNED MODEL CORRECTION METRICS
# =============================================================================

logger.info("\n" + "="*80)
logger.info("  SECTION 4: FINE-TUNED MODEL CORRECTION METRICS")
logger.info("="*80)

MODEL_FILE = os.path.join(BASE_DIR, args.model_corrections)

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE) as f:
        model_data = json.load(f)
    
    # Handle both formats
    if isinstance(model_data, dict) and "results" in model_data:
        results = model_data["results"]
        summary = model_data.get("summary", {})
    else:
        results = model_data
        summary = {}
    
    total_examples = len(results)
    garbage_count = sum(1 for r in results if r.get('is_garbage', False))
    clean_results = [r for r in results if not r.get('is_garbage', False)]
    
    logger.info(f"\n{'─'*60}")
    logger.info(f"  OUTPUT QUALITY")
    logger.info(f"{'─'*60}")
    logger.info(f"  Total Examples Tested:      {total_examples}")
    logger.info(f"  Clean Outputs:              {len(clean_results)}  ({len(clean_results)/total_examples*100:.1f}%)")
    logger.info(f"  Garbage Outputs:            {garbage_count}  ({garbage_count/total_examples*100:.1f}%)")
    
    if clean_results:
        # Hallucination removal
        if 'removal_rate' in clean_results[0]:
            avg_removal = np.mean([r['removal_rate'] for r in clean_results])
            full_removal = sum(1 for r in clean_results if r['removal_rate'] == 100)
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"  HALLUCINATION CORRECTION")
            logger.info(f"{'─'*60}")
            logger.info(f"  Avg Removal Rate:           {avg_removal:6.1f}%")
            logger.info(f"  Full Removal (100%):        {full_removal}/{len(clean_results)}  ({full_removal/len(clean_results)*100:.1f}%)")
        
        # Similarity to human
        if 'sim_to_human' in clean_results[0]:
            avg_sim_human = np.mean([r['sim_to_human'] for r in clean_results])
            avg_sim_orig = np.mean([r['sim_to_original'] for r in clean_results])
            high_sim = sum(1 for r in clean_results if r['sim_to_human'] > 0.5)
            
            logger.info(f"\n{'─'*60}")
            logger.info(f"  SIMILARITY METRICS")
            logger.info(f"{'─'*60}")
            logger.info(f"  Avg Similarity to Human:    {avg_sim_human:.3f}")
            logger.info(f"  Avg Similarity to Original: {avg_sim_orig:.3f}")
            logger.info(f"  High Similarity (>0.5):     {high_sim}/{len(clean_results)}  ({high_sim/len(clean_results)*100:.1f}%)")
            
            # Semantic change indicator
            semantic_change = avg_sim_orig - avg_sim_human
            logger.info(f"  Semantic Shift:             {abs(semantic_change):.3f}  ({'toward human' if semantic_change > 0 else 'minimal change'})")

else:
    logger.warning(f"Model corrections file not found: {MODEL_FILE}")

# =============================================================================
# SECTION 5: END-TO-END PIPELINE SUMMARY
# =============================================================================

logger.info("\n" + "="*80)
logger.info("  SECTION 5: END-TO-END PIPELINE SUMMARY")
logger.info("="*80)

logger.info(f"\n{'─'*60}")
logger.info(f"  PIPELINE COMPONENTS")
logger.info(f"{'─'*60}")

# Component status
components = {
    "Image Captioning": "BLIP-2 / LLaVA",
    "Object Detection": "OWL-ViT",
    "Color Verification": "Region-based extraction",
    "Human Annotation": f"{len(human_data) if human_data else 0} images",
    "Fine-tuning Data": f"{len(gold_data) if 'gold_data' in dir() else 0} examples",
    "Correction Model": "Qwen2.5-7B + LoRA"
}

for component, status in components.items():
    logger.info(f"  {component:25s}  {status}")

logger.info(f"\n{'─'*60}")
logger.info(f"  KEY PERFORMANCE INDICATORS")
logger.info(f"{'─'*60}")

# Collect all KPIs
kpis = []

if 'chair_i' in dir():
    kpis.append(("CHAIR_i (Detection)", f"{chair_i:.1%}", "Lower is better"))
if 'chair_s' in dir():
    kpis.append(("CHAIR_s (Detection)", f"{chair_s:.1%}", "Lower is better"))
if 'accuracy' in dir() and y_true:
    kpis.append(("AI-Human Agreement", f"{accuracy:.1%}", "Higher is better"))
if 'f1' in dir() and y_true:
    kpis.append(("F1 Score (Agreement)", f"{f1:.1%}", "Higher is better"))
if 'kappa' in dir() and y_true:
    kpis.append(("Cohen's Kappa", f"{kappa:.3f}", "Higher is better"))
if 'clean_results' in dir() and clean_results:
    kpis.append(("Clean Outputs", f"{len(clean_results)/total_examples:.1%}", "Higher is better"))
    if 'avg_removal' in dir():
        kpis.append(("Hallucination Removal", f"{avg_removal:.1f}%", "Higher is better"))
    if 'avg_sim_human' in dir():
        kpis.append(("Similarity to Human", f"{avg_sim_human:.3f}", "Higher is better"))

for metric, value, note in kpis:
    logger.info(f"  {metric:28s}  {value:>10s}  ({note})")

# =============================================================================
# SECTION 6: RECOMMENDATIONS
# =============================================================================

logger.info("\n" + "="*80)
logger.info("  SECTION 6: RECOMMENDATIONS")
logger.info("="*80)

recommendations = []

# Based on metrics
if 'chair_s' in dir() and chair_s > 0.30:
    recommendations.append("• High CHAIR_s: Consider improving captioner or detection threshold")
elif 'chair_s' in dir() and chair_s < 0.15:
    recommendations.append("• Good CHAIR_s: Detection pipeline working well")

if 'kappa' in dir() and kappa < 0.4:
    recommendations.append("• Low Cohen's Kappa: More human annotations needed for calibration")
elif 'kappa' in dir() and kappa > 0.6:
    recommendations.append("• Strong agreement: AI detection aligns well with human judgment")

if 'garbage_count' in dir() and garbage_count == 0:
    recommendations.append("• No garbage outputs: Model fine-tuning successful")
elif 'garbage_count' in dir() and garbage_count > 0:
    recommendations.append("• Garbage outputs present: Review training configuration")

if 'avg_removal' in dir() and avg_removal < 30:
    recommendations.append("• Low removal rate: Model is conservative, may need more training data")
elif 'avg_removal' in dir() and avg_removal > 50:
    recommendations.append("• Good removal rate: Model effectively addresses hallucinations")

if 'gold_data' in dir() and len(gold_data) < 200:
    recommendations.append("• Small training set: Collect more human annotations for better results")

logger.info("")
for rec in recommendations:
    logger.info(f"  {rec}")

# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

logger.info("\n" + "="*80)
logger.info("  FINAL SUMMARY TABLE")
logger.info("="*80)

logger.info("""
┌─────────────────────────────────────────────────────────────────────┐
│                    PEEKABOO METRICS SUMMARY                        │
├─────────────────────────────────────────────────────────────────────┤
│  DETECTION PIPELINE                                                 │""")

if 'chair_i' in dir():
    logger.info(f"│    CHAIR_i:                    {chair_i:>6.1%}                            │")
if 'chair_s' in dir():
    logger.info(f"│    CHAIR_s:                    {chair_s:>6.1%}                            │")
if 'detection_coverage' in dir():
    logger.info(f"│    Detection Coverage:         {detection_coverage:>6.1%}                            │")

logger.info("├─────────────────────────────────────────────────────────────────────┤")
logger.info("│  HUMAN AGREEMENT                                                    │")

if 'accuracy' in dir() and y_true:
    logger.info(f"│    Accuracy:                   {accuracy:>6.1%}                            │")
    logger.info(f"│    Precision:                  {precision:>6.1%}                            │")
    logger.info(f"│    Recall:                     {recall:>6.1%}                            │")
    logger.info(f"│    F1 Score:                   {f1:>6.1%}                            │")
    logger.info(f"│    Cohen's Kappa:              {kappa:>6.3f}                            │")

logger.info("├─────────────────────────────────────────────────────────────────────┤")
logger.info("│  MODEL CORRECTION                                                   │")

if 'clean_results' in dir() and clean_results:
    clean_pct = len(clean_results)/total_examples*100
    logger.info(f"│    Clean Outputs:              {clean_pct:>5.1f}%                            │")
    if 'avg_removal' in dir():
        logger.info(f"│    Avg Hallucination Removal:  {avg_removal:>5.1f}%                            │")
    if 'avg_sim_human' in dir():
        logger.info(f"│    Similarity to Human:        {avg_sim_human:>5.3f}                            │")

logger.info("└─────────────────────────────────────────────────────────────────────┘")

# --- FOOTER ---
logger.info("\n" + "="*80)
logger.info(f"  Report saved to: {log_filename}")
logger.info(f"  Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)