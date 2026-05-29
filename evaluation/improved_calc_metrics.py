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
log_filename = f"logs/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
parser.add_argument("--file", type=str, default="data/output/results.json",
                    help="Path to results.json from pipeline")
parser.add_argument("--confidence_threshold", type=float, default=0.7,
                    help="Minimum confidence to count as hallucination (0-1)")
parser.add_argument("--human_file", type=str, default="data/output/human_annotations.json",
                    help="Path to human annotations for agreement metrics")
args = parser.parse_args()

# --- HEADER ---
logger.info("="*70)
logger.info("           PEEKABOO HALLUCINATION METRICS REPORT")
logger.info("="*70)
logger.info(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Results File: {args.file}")
logger.info(f"Confidence Threshold: {args.confidence_threshold}")
logger.info("="*70)

# --- LOAD DATA ---
if not os.path.exists(args.file):
    logger.error(f"Results file not found: {args.file}")
    exit(1)

with open(args.file) as f:
    data = json.load(f)

logger.info(f"\nDataset Overview:")
logger.info(f"   Total Images Processed: {len(data)}")

# --- INITIALIZE COUNTERS ---
# CHAIR Metrics
images_with_hallucination = 0
total_hallucinated_objects = 0
total_mentioned_objects = 0

# Hallucination Type Breakdown
hallucination_types = defaultdict(int)
confidence_scores = []

# Detection Statistics
detection_stats = {
    "total_claims": 0,
    "claims_with_detections": 0,
    "total_detections": 0,
    "avg_detection_score": []
}

# Color Verification Stats
color_stats = {
    "total_color_claims": 0,
    "color_matches": 0,
    "color_mismatches": 0
}

# Count Verification Stats
count_stats = {
    "total_count_claims": 0,
    "exact_matches": 0,
    "within_tolerance": 0,
    "mismatches": 0
}

# Per-Image Statistics
per_image_stats = []

# --- PROCESS EACH IMAGE ---
logger.info("\nProcessing Verification Results...")

for idx, item in enumerate(data):
    image_id = item['image_id']
    caption = item['caption']
    verifications = item['verification']
    
    has_hallucination_in_image = False
    image_hall_types = []
    image_confidences = []
    
    for v in verifications:
        total_mentioned_objects += 1
        detection_stats["total_claims"] += 1
        
        h_type = v['hallucination_type']
        confidence = v.get('confidence', 0)
        detected_count = v['detected_count']
        expected_count = v.get('expected_count', 1)
        
        # Track detection statistics
        if detected_count > 0:
            detection_stats["claims_with_detections"] += 1
            detection_stats["total_detections"] += detected_count
            
            # Track detection scores
            if 'detection_scores' in v and v['detection_scores']:
                detection_stats["avg_detection_score"].extend(v['detection_scores'])
        
        # Color verification tracking
        if v.get('expected_color'):
            color_stats["total_color_claims"] += 1
            if v.get('color_match') is True:
                color_stats["color_matches"] += 1
            elif v.get('color_match') is False:
                color_stats["color_mismatches"] += 1
        
        # Count verification tracking
        if expected_count > 1:
            count_stats["total_count_claims"] += 1
            if detected_count == expected_count:
                count_stats["exact_matches"] += 1
            elif abs(detected_count - expected_count) <= 1:
                count_stats["within_tolerance"] += 1
            else:
                count_stats["mismatches"] += 1
        
        # Hallucination detection (using confidence threshold)
        if h_type != "None" and confidence >= args.confidence_threshold:
            total_hallucinated_objects += 1
            has_hallucination_in_image = True
            hallucination_types[h_type] += 1
            confidence_scores.append(confidence)
            image_hall_types.append(h_type)
            image_confidences.append(confidence)
    
    if has_hallucination_in_image:
        images_with_hallucination += 1
    
    # Store per-image stats
    per_image_stats.append({
        "image_id": image_id,
        "caption_length": len(caption.split()),
        "num_claims": len(verifications),
        "num_hallucinations": len(image_hall_types),
        "hallucination_types": image_hall_types,
        "avg_confidence": np.mean(image_confidences) if image_confidences else 0
    })

# --- CALCULATE METRICS ---
chair_i = images_with_hallucination / len(data) if data else 0
chair_s = total_hallucinated_objects / total_mentioned_objects if total_mentioned_objects else 0

# Precision/Recall approximation (treating all mentions as positive claims)
precision = (total_mentioned_objects - total_hallucinated_objects) / total_mentioned_objects if total_mentioned_objects else 0
recall_proxy = detection_stats["claims_with_detections"] / detection_stats["total_claims"] if detection_stats["total_claims"] else 0

# Average confidence
avg_confidence = np.mean(confidence_scores) if confidence_scores else 0

# Average detection score
avg_detection_score = np.mean(detection_stats["avg_detection_score"]) if detection_stats["avg_detection_score"] else 0

# Color accuracy
color_accuracy = color_stats["color_matches"] / color_stats["total_color_claims"] if color_stats["total_color_claims"] else 0

# Count accuracy
count_accuracy = (count_stats["exact_matches"] + count_stats["within_tolerance"]) / count_stats["total_count_claims"] if count_stats["total_count_claims"] else 0

# --- REPORT GENERATION ---
logger.info("\n" + "="*70)
logger.info("                    CORE METRICS SUMMARY")
logger.info("="*70)

logger.info("\nCHAIR Metrics (Lower is Better)")
logger.info("-" * 70)
logger.info(f"   CHAIR_i (Image-level Error Rate):     {chair_i:6.2%}  {'Good' if chair_i < 0.30 else 'High' if chair_i < 0.50 else 'Very High'}")
logger.info(f"   CHAIR_s (Sentence-level Error Rate):  {chair_s:6.2%}  {'Good' if chair_s < 0.15 else 'High' if chair_s < 0.30 else 'Very High'}")
logger.info(f"   Images with Hallucinations:           {images_with_hallucination}/{len(data)}")
logger.info(f"   Total Hallucinated Objects:           {total_hallucinated_objects}/{total_mentioned_objects}")

logger.info("\nFaithfulness Metrics (Higher is Better)")
logger.info("-" * 70)
logger.info(f"   Caption Precision (Non-hallucinated): {precision:6.2%}")
logger.info(f"   Detection Coverage (Recall Proxy):    {recall_proxy:6.2%}")
logger.info(f"   Average Hallucination Confidence:     {avg_confidence:6.2f}/1.00")

logger.info("\nDetection Statistics")
logger.info("-" * 70)
logger.info(f"   Total Claims Verified:                {detection_stats['total_claims']}")
logger.info(f"   Claims with Detections:               {detection_stats['claims_with_detections']} ({detection_stats['claims_with_detections']/detection_stats['total_claims']*100:.1f}%)")
logger.info(f"   Total Object Detections:              {detection_stats['total_detections']}")
logger.info(f"   Average Detection Score:              {avg_detection_score:.3f}")

logger.info("\nAttribute Verification (Color)")
logger.info("-" * 70)
if color_stats["total_color_claims"] > 0:
    logger.info(f"   Total Color Claims:                   {color_stats['total_color_claims']}")
    logger.info(f"   Color Matches:                        {color_stats['color_matches']} ({color_accuracy*100:.1f}%)")
    logger.info(f"   Color Mismatches:                     {color_stats['color_mismatches']}")
    logger.info(f"   Color Verification Accuracy:          {color_accuracy:6.2%}  {'Good' if color_accuracy > 0.70 else 'Moderate' if color_accuracy > 0.50 else 'Poor'}")
else:
    logger.info(f"   No color claims found in captions")

logger.info("\nCount Verification")
logger.info("-" * 70)
if count_stats["total_count_claims"] > 0:
    logger.info(f"   Total Count Claims (>1):              {count_stats['total_count_claims']}")
    logger.info(f"   Exact Matches:                        {count_stats['exact_matches']}")
    logger.info(f"   Within ±1 Tolerance:                  {count_stats['within_tolerance']}")
    logger.info(f"   Count Mismatches:                     {count_stats['mismatches']}")
    logger.info(f"   Count Accuracy (with tolerance):      {count_accuracy:6.2%}  {'Good' if count_accuracy > 0.60 else 'Moderate' if count_accuracy > 0.40 else 'Poor'}")
else:
    logger.info(f"   No explicit count claims (>1) in captions")

logger.info("\nHallucination Type Breakdown")
logger.info("-" * 70)
if hallucination_types:
    for h_type, count in sorted(hallucination_types.items(), key=lambda x: x[1], reverse=True):
        percentage = count / total_hallucinated_objects * 100 if total_hallucinated_objects > 0 else 0
        logger.info(f"   {h_type:25s}  {count:4d} ({percentage:5.1f}%)")
else:
    logger.info(f"   No hallucinations detected")

# --- PER-IMAGE ANALYSIS ---
logger.info("\n" + "="*70)
logger.info("                 DETAILED PER-IMAGE ANALYSIS")
logger.info("="*70)

# Sort by number of hallucinations
sorted_images = sorted(per_image_stats, key=lambda x: x['num_hallucinations'], reverse=True)

logger.info(f"\nTop 10 Images with Most Hallucinations:")
logger.info("-" * 70)
for i, img_stat in enumerate(sorted_images[:10], 1):
    logger.info(f"{i:2d}. {img_stat['image_id']:30s}  "
                f"Hall: {img_stat['num_hallucinations']}/{img_stat['num_claims']}  "
                f"Conf: {img_stat['avg_confidence']:.2f}  "
                f"Types: {', '.join(set(img_stat['hallucination_types']))}")

logger.info(f"\nImages with No Hallucinations: {len(data) - images_with_hallucination}")

# --- HUMAN AGREEMENT ANALYSIS ---
if os.path.exists(args.human_file):
    logger.info("\n" + "="*70)
    logger.info("              HUMAN AGREEMENT ANALYSIS")
    logger.info("="*70)
    
    with open(args.human_file) as f:
        human_data = json.load(f)
    
    logger.info(f"\nHuman Annotations Available: {len(human_data)} images")
    
    # Match AI and human annotations
    ai_dict = {item['image_id']: item for item in data}
    
    agreement_count = 0
    total_comparisons = 0
    disagreements = []
    
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
            
            total_comparisons += 1
            
            # AI prediction (using confidence threshold)
            ai_hall = v['hallucination_type'] != "None" and v.get('confidence', 0) >= args.confidence_threshold
            human_hall = human_labels[claim] == "Hallucination"
            
            if ai_hall == human_hall:
                agreement_count += 1
            else:
                disagreements.append({
                    "image_id": img_id,
                    "claim": claim,
                    "ai_verdict": v['hallucination_type'],
                    "ai_confidence": v.get('confidence', 0),
                    "human_verdict": human_labels[claim]
                })
    
    if total_comparisons > 0:
        agreement_rate = agreement_count / total_comparisons
        logger.info(f"\nAgreement Metrics:")
        logger.info("-" * 70)
        logger.info(f"   Total Comparisons:                    {total_comparisons}")
        logger.info(f"   Agreements:                           {agreement_count}")
        logger.info(f"   Disagreements:                        {len(disagreements)}")
        logger.info(f"   Agreement Rate:                       {agreement_rate:6.2%}  {'Strong' if agreement_rate > 0.80 else ' Moderate' if agreement_rate > 0.60 else 'Weak'}")
        
        if disagreements and len(disagreements) <= 20:
            logger.info(f"\nSample Disagreements:")
            logger.info("-" * 70)
            for i, dis in enumerate(disagreements[:10], 1):
                logger.info(f"{i:2d}. Image: {dis['image_id']}")
                logger.info(f"    Claim: '{dis['claim']}'")
                logger.info(f"    AI: {dis['ai_verdict']} (conf: {dis['ai_confidence']:.2f}) | Human: {dis['human_verdict']}")
        
        # Calculate Precision, Recall, F1 against human labels
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        y_true = []
        y_pred = []
        
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
                
                ai_hall = 1 if (v['hallucination_type'] != "None" and v.get('confidence', 0) >= args.confidence_threshold) else 0
                human_hall = 1 if human_labels[claim] == "Hallucination" else 0
                
                y_true.append(human_hall)
                y_pred.append(ai_hall)
        
        if y_true and y_pred:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            logger.info(f"\nClassification Metrics vs Human Labels:")
            logger.info("-" * 70)
            logger.info(f"   Precision:                            {prec:6.2%}")
            logger.info(f"   Recall:                               {rec:6.2%}")
            logger.info(f"   F1 Score:                             {f1:6.2%}")
    else:
        logger.info(f"\nNo overlapping annotations found for comparison")
else:
    logger.info(f"\nHuman annotation file not found: {args.human_file}")
    logger.info(f"   Run annotation UI to collect human judgments for agreement analysis")

# --- RECOMMENDATIONS ---
logger.info("\n" + "="*70)
logger.info("                   RECOMMENDATIONS")
logger.info("="*70)

recommendations = []

if chair_s > 0.30:
    recommendations.append("CHAIR_s is high (>30%). Consider fine-tuning captioner or adjusting threshold.")
elif chair_s > 0.15:
    recommendations.append("CHAIR_s is moderate. Review detection pipeline configuration.")
else:
    recommendations.append("CHAIR_s is good (<15%). Caption quality is reasonable.")

if avg_detection_score < 0.10:
    recommendations.append("Low average detection scores. Consider lowering confidence threshold.")

if color_stats["total_color_claims"] > 0 and color_accuracy < 0.60:
    recommendations.append("Color verification accuracy is low. Review color extraction logic.")

if count_stats["total_count_claims"] > 0 and count_accuracy < 0.50:
    recommendations.append("Count verification is weak. OWL-ViT may struggle with counting.")

if detection_stats["claims_with_detections"] / detection_stats["total_claims"] < 0.60:
    recommendations.append("Many objects not detected. Lower threshold or use ensemble methods.")

if not recommendations:
    recommendations.append("All metrics look good. Pipeline is working well.")

logger.info("")
for rec in recommendations:
    logger.info(f"   {rec}")

# --- FOOTER ---
logger.info("\n" + "="*70)
logger.info(f"Full report saved to: {log_filename}")
logger.info(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*70)