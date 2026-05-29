import torch
import spacy
import os
import json
import argparse
import logging
import sys
import re
from datetime import datetime
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np

# LOGGING SETUP
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# CONFIG
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, default="data/images")
parser.add_argument("--output_file", type=str, default="data/output/results.json")
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--confidence_threshold", type=float, default=0.15)
parser.add_argument("--count_tolerance", type=int, default=1)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("="*80)
logger.info("           PEEKABOO HALLUCINATION DETECTION PIPELINE")
logger.info("="*80)
logger.info(f"Pipeline Version: 2.1 (OWLv2 + FIXED COLOR VERIFICATION)")
logger.info(f"Device: {DEVICE}")
logger.info(f"Images Directory: {args.image_dir}")
logger.info(f"Processing Limit: {args.limit}")
logger.info(f"Detection Confidence Threshold: {args.confidence_threshold}")
logger.info(f"Count Tolerance: ±{args.count_tolerance}")
logger.info("="*80)

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

# Load Spacy
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Spacy model loaded successfully")
except:
    logger.warning("Spacy model not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# MODEL LOADING
logger.info("\n" + "-"*80)
logger.info("Loading BLIP-2 (Image Captioner)...")
try:
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    logger.info("BLIP-2 loaded successfully")
except Exception as e:
    logger.error(f"Failed to load BLIP-2: {e}")
    exit(1)

logger.info("\nLoading OWLv2 (Object Detector - Open World)...")
try:
    owl_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owl_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE)
    owl_model.eval()
    logger.info("OWLv2 loaded successfully")
    logger.info("OWLv2 Advantages: Better generalization, lower false positives, ensemble features")
except Exception as e:
    logger.error(f"Failed to load OWLv2: {e}")
    exit(1)

logger.info("-"*80)

# HELPER FUNCTIONS
def get_caption(image):
    """Generate caption using BLIP-2"""
    try:
        inputs = blip_processor(images=image, return_tensors="pt").to(blip_model.device, torch.float16)
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        logger.debug(f"Generated caption: {caption}")
        return caption
    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        return "Error generating caption"

def extract_claims(caption):
    """
    Enhanced claim extraction with aggressive generic term filtering
    FIXED: Color extraction now only extracts colors that modify nouns
    """
    doc = nlp(caption)
    claims = []
    
    # Terms that OWLv2 cannot reliably detect
    SKIP_TERMS = {
        # Abstract locations and spatial terms
        'field', 'road', 'area', 'scene', 'background', 'ground', 'place', 'spot',
        'way', 'side', 'end', 'top', 'bottom', 'front', 'back', 'center', 'middle',
        'space', 'location', 'position', 'corner', 'edge', 'surface',
        
        # Abstract groupings
        'group', 'bunch', 'collection', 'set', 'pair', 'lot', 'number', 'amount',
        
        # Abstract concepts
        'thing', 'something', 'part', 'piece', 'section', 'portion', 'element',
        
        # Pronouns and vague references
        'it', 'its', 'them', 'their', 'one', 'ones', 'other', 'others',
        
        # Materials and textures (hard to detect)
        'hay', 'grass', 'dirt', 'sand', 'water', 'snow', 'ice', 'mud', 'dust',
        'wood', 'metal', 'plastic', 'fabric', 'leather', 'paper',
        
        # Environmental conditions
        'sunlight', 'shadow', 'light', 'darkness', 'shade', 'sunshine', 'rain',
        'weather', 'sky', 'cloud', 'sun', 'moon',
        
        # Abstract attributes
        'view', 'appearance', 'look', 'style', 'design', 'pattern', 'texture',
        
        # Time references
        'day', 'night', 'time', 'moment', 'season', 'morning', 'evening'
    }
    
    # Body parts (only skip if not main subject)
    BODY_PARTS = {
        'mouth', 'eye', 'eyes', 'nose', 'ear', 'ears', 'hand', 'hands', 'leg', 'legs',
        'arm', 'arms', 'foot', 'feet', 'head', 'tail', 'wing', 'wings', 'paw', 'paws',
        'face', 'body', 'neck', 'back', 'stomach', 'chest'
    }
    
    # Number word mapping
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1, "several": 3, "few": 2, "couple": 2,
        "many": 5, "some": 2, "multiple": 3
    }
    
    # Color list
    colors = ["red", "blue", "green", "yellow", "black", "white", 
              "orange", "purple", "brown", "pink", "gray", "grey", 
              "silver", "gold", "beige", "tan", "violet", "cyan", "magenta"]
    
    filtered_count = 0
    
    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        root = chunk.root.text.lower()
        
        # Filter 1: Skip generic/abstract terms
        if root in SKIP_TERMS:
            filtered_count += 1
            logger.debug(f"Filtered generic term: '{chunk.text}'")
            continue
        
        # Filter 2: Skip pronouns
        if chunk.root.pos_ == "PRON":
            filtered_count += 1
            logger.debug(f"Filtered pronoun: '{chunk.text}'")
            continue
        
        # Filter 3: Skip body parts unless they're standalone subject
        if root in BODY_PARTS and len(chunk) < 3:
            filtered_count += 1
            logger.debug(f"Filtered body part: '{chunk.text}'")
            continue
        
        # Filter 4: Skip very short determiners
        if len(chunk.text.split()) == 1 and root in ['a', 'an', 'the']:
            filtered_count += 1
            continue
        
        # Filter 5: Skip if the root is too short (likely determiner)
        if len(root) <= 2:
            filtered_count += 1
            continue
        
        # Extract count
        count = 1
        count_word = None
        for token in chunk:
            if token.pos_ == "NUM":
                if token.text.isdigit():
                    count = int(token.text)
                    count_word = token.text
                elif token.text.lower() in number_words:
                    count = number_words[token.text.lower()]
                    count_word = token.text.lower()
        
        # FIXED: Extract color (only if it's a modifier before the noun)
        color = None
        
        # Special case: color-compound nouns (bluebell, orange fruit)
        color_compounds = {
            "bluebell": "blue", 
            "bluebells": "blue",
            "blackberry": "black", 
            "blackberries": "black",
            "orange": "orange",  # the fruit
            "oranges": "orange"
        }
        
        if root in color_compounds:
            color = color_compounds[root]
        else:
            # Only extract if color word appears BEFORE the root noun
            for col in colors:
                # Use word boundary matching
                pattern = r'\b' + col + r'\b'
                if re.search(pattern, text):
                    # Check position: color should come before noun
                    color_pos = text.find(col)
                    noun_pos = text.find(root)
                    if color_pos < noun_pos:
                        color = col
                        break
        
        claims.append({
            "phrase": root,
            "full_text": chunk.text,
            "exp_count": count,
            "exp_color": color,
            "count_word": count_word,
            "has_explicit_count": count_word is not None
        })
    
    if filtered_count > 0:
        logger.debug(f"Filtered {filtered_count} generic/abstract claims from: '{caption}'")
    
    return claims

def calibrate_confidence(raw_score, detection_type="existence"):
    """
    Calibrate raw detection scores to better confidence estimates
    OWLv2 scores tend to be more reliable than OWL-ViT
    """
    if detection_type == "existence":
        # OWLv2 is good at existence detection
        if raw_score < 0.10:
            return 0.3  # Low confidence
        elif raw_score < 0.20:
            return 0.6  # Medium confidence
        else:
            return 0.9  # High confidence
    
    elif detection_type == "count":
        # Count is inherently harder
        if raw_score < 0.15:
            return 0.4
        elif raw_score < 0.25:
            return 0.7
        else:
            return 0.85
    
    elif detection_type == "color":
        # Color verification has its own logic
        return 0.8
    
    return raw_score

def verify_single_object(image, object_name, threshold=0.15):
    """
    Verify a single object using OWLv2
    Returns: boxes, scores
    """
    try:
        inputs = owl_processor(text=[[object_name]], images=image, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = owl_model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]]).to(DEVICE)
        results = owl_processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        
        boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.array([])
        scores = results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([])
        
        logger.debug(f"Object '{object_name}': Found {len(boxes)} detections, "
                    f"scores: {[f'{s:.3f}' for s in scores[:3]]}")
        
        return boxes, scores
        
    except Exception as e:
        logger.error(f"Detection failed for '{object_name}': {e}")
        return np.array([]), np.array([])

def extract_color_from_region(image, box):
    """
    IMPROVED: Extract dominant color from bounding box region
    Uses dominant color (mode) instead of mean for better accuracy
    Better thresholds tuned for real-world images with various lighting
    """
    try:
        x1, y1, x2, y2 = [int(coord) for coord in box]
        
        # Ensure valid box coordinates
        x1, x2 = max(0, x1), min(image.size[0], x2)
        y1, y2 = max(0, y1), min(image.size[1], y2)
        
        if x2 <= x1 or y2 <= y1:
            return "unknown"
        
        region = image.crop((x1, y1, x2, y2))
        region_array = np.array(region)
        
        # Get dominant color (more robust than mean)
        pixels = region_array.reshape(-1, 3)
        quantized = (pixels // 32) * 32  # Quantize to reduce noise
        unique_colors, counts = np.unique(quantized, axis=0, return_counts=True)
        
        if len(unique_colors) > 0:
            dominant_color = unique_colors[counts.argmax()]
        else:
            dominant_color = region_array.mean(axis=(0, 1))
        
        r, g, b = dominant_color
        
        # IMPROVED: Better color classification thresholds
        # White (bright, balanced)
        if r > 180 and g > 180 and b > 180:
            return "white"
        
        # Black (dark, balanced)
        if r < 50 and g < 50 and b < 50:
            return "black"
        
        # Gray (neutral, balanced)
        if abs(r - g) < 40 and abs(g - b) < 40 and abs(r - b) < 40:
            if 50 <= r <= 180:
                return "gray"
        
        # Chromatic colors (use relative ratios for better lighting handling)
        # Red (R dominant)
        if r > 100 and r > g * 1.3 and r > b * 1.3:
            return "red"
        
        # Blue (B dominant)
        if b > 100 and b > r * 1.2 and b > g * 1.2:
            return "blue"
        
        # Green (G dominant)
        if g > 100 and g > r * 1.2 and g > b * 1.2:
            return "green"
        
        # Yellow (R and G high, B low)
        if r > 150 and g > 150 and b < 120:
            return "yellow"
        
        # Orange (R high, G medium, B low)
        if r > 150 and 80 < g < 200 and b < 100 and r > g * 1.1:
            return "orange"
        
        # Purple (R and B high, G low)
        if r > 80 and b > 80 and g < r * 0.8:
            return "purple"
        
        # Pink (R high, G and B medium-high)
        if r > 150 and 100 < g < 180 and 100 < b < 180 and r > g * 1.1:
            return "pink"
        
        # Brown (darker orange-red)
        if 80 < r < 180 and 50 < g < 130 and 30 < b < 100:
            return "brown"
        
        return "unknown"
        
    except Exception as e:
        logger.error(f"Color extraction failed: {e}")
        return "unknown"

def verify_claims_improved(image, claims, threshold=0.15, count_tolerance=1):
    """
    Improved verification with OWLv2:
    1. Individual object detection for each claim
    2. FIXED: Attribute verification (color) with better matching
    3. Smart count matching with tolerance
    4. Calibrated confidence scores
    """
    results = []
    if not claims:
        return []
    
    for claim in claims:
        object_name = claim['phrase']
        expected_count = claim['exp_count']
        expected_color = claim['exp_color']
        has_explicit_count = claim['has_explicit_count']
        
        # Detect object using OWLv2
        boxes, scores = verify_single_object(image, object_name, threshold)
        detected_count = len(boxes)
        
        # Initialize result
        h_type = "None"
        raw_confidence = 0.0
        calibrated_confidence = 0.0
        color_match = None
        detected_colors = []
        
        # Check 1: Object Existence
        if detected_count == 0:
            h_type = "Object Existence"
            raw_confidence = 1.0
            calibrated_confidence = calibrate_confidence(1.0, "existence")
        
        # Check 2: Count Verification (only if explicit count mentioned)
        elif has_explicit_count and expected_count > 1:
            count_diff = abs(detected_count - expected_count)
            
            if count_diff > count_tolerance:
                h_type = "Count"
                raw_confidence = min(1.0, count_diff / expected_count)
                calibrated_confidence = calibrate_confidence(raw_confidence, "count")
        
        # Check 3: FIXED Color Verification (if object exists and color mentioned)
        if detected_count > 0 and expected_color:
            for box in boxes:
                detected_color = extract_color_from_region(image, box)
                detected_colors.append(detected_color)
            
            # FIXED: Check if any detected box has the expected color
            # Allow for color synonyms (orange/red, gray/grey)
            color_synonyms = {
                "gray": ["grey"],
                "grey": ["gray"],
                "orange": ["red"],  # Oranges often appear reddish
            }
            
            match_count = detected_colors.count(expected_color)
            if expected_color in color_synonyms:
                for synonym in color_synonyms[expected_color]:
                    match_count += detected_colors.count(synonym)
            
            # Color matches if at least 50% of detections match
            total_detections = len(detected_colors)
            color_match = (match_count / total_detections >= 0.5) if total_detections > 0 else False
            
            if not color_match and h_type == "None":
                h_type = "Attribute (Color)"
                raw_confidence = 0.8
                calibrated_confidence = calibrate_confidence(raw_confidence, "color")
        
        # If no hallucination detected, calculate faithfulness confidence
        if h_type == "None" and len(scores) > 0:
            # Average of top detection scores as faithfulness measure
            faithfulness_score = float(np.mean(scores))
            calibrated_confidence = min(0.95, faithfulness_score * 5)  # Scale to 0-0.95
        
        results.append({
            "claim": claim['full_text'],
            "hallucination_type": h_type,
            "confidence": round(calibrated_confidence, 3),
            "raw_confidence": round(raw_confidence, 3),
            "detected_count": detected_count,
            "expected_count": expected_count,
            "detection_scores": [round(float(s), 3) for s in scores] if len(scores) > 0 else [],
            "expected_color": expected_color,
            "detected_colors": detected_colors if expected_color else None,
            "color_match": color_match if expected_color else None
        })
    
    return results

# MAIN PROCESSING LOOP
if not os.path.exists(args.image_dir):
    logger.error(f"Image directory '{args.image_dir}' does not exist.")
    exit(1)

images_list = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
logger.info(f"\nFound {len(images_list)} images in directory")
logger.info(f"Processing limit: {min(len(images_list), args.limit)} images")
logger.info("\n" + "-"*80)
logger.info("Starting Image Processing...")
logger.info("-"*80)

# Statistics tracking
stats = {
    "total_processed": 0,
    "total_captions_generated": 0,
    "total_claims_extracted": 0,
    "claims_filtered": 0,
    "claims_verified": 0,
    "images_with_hallucinations": 0,
    "total_hallucinations": 0,
    "color_claims": 0,
    "processing_errors": 0
}

data = []
error_images = []

for i, img_name in enumerate(images_list[:args.limit]):
    if i % 10 == 0:
        logger.info(f"Progress: {i}/{min(len(images_list), args.limit)} images processed...")
    
    try:
        path = os.path.join(args.image_dir, img_name)
        image = Image.open(path).convert("RGB")
        
        # Step 1: Generate caption
        cap = get_caption(image)
        stats["total_captions_generated"] += 1
        
        # Step 2: Extract claims with filtering
        all_chunks = len(list(nlp(cap).noun_chunks))
        claims = extract_claims(cap)
        stats["claims_filtered"] += (all_chunks - len(claims))
        stats["total_claims_extracted"] += len(claims)
        
        # Count color claims
        stats["color_claims"] += sum(1 for c in claims if c['exp_color'] is not None)
        
        # Step 3: Verify claims
        verification = verify_claims_improved(image, claims, 
                                             threshold=args.confidence_threshold,
                                             count_tolerance=args.count_tolerance)
        stats["claims_verified"] += len(verification)
        
        # Count hallucinations
        hallucinations = [v for v in verification if v['hallucination_type'] != "None"]
        if hallucinations:
            stats["images_with_hallucinations"] += 1
            stats["total_hallucinations"] += len(hallucinations)
        
        data.append({
            "image_id": img_name,
            "caption": cap,
            "verification": verification,
            "processing_timestamp": datetime.now().isoformat()
        })
        
        stats["total_processed"] += 1
        
    except Exception as e:
        logger.error(f"Error processing {img_name}: {e}")
        error_images.append({"image_id": img_name, "error": str(e)})
        stats["processing_errors"] += 1

# Save results
logger.info("\n" + "-"*80)
logger.info("Saving Results...")
logger.info("-"*80)

try:
    with open(args.output_file, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Results saved to: {args.output_file}")
except Exception as e:
    logger.error(f"Failed to save results: {e}")

# Log statistics
logger.info("\n" + "="*80)
logger.info("                  PIPELINE STATISTICS")
logger.info("="*80)

logger.info(f"\nProcessing Summary:")
logger.info(f"   Images Successfully Processed: {stats['total_processed']}/{min(len(images_list), args.limit)}")
logger.info(f"   Processing Errors: {stats['processing_errors']}")
logger.info(f"   Captions Generated: {stats['total_captions_generated']}")

logger.info(f"\nClaim Extraction:")
logger.info(f"   Total Claims Extracted: {stats['total_claims_extracted']}")
logger.info(f"   Generic Claims Filtered: {stats['claims_filtered']}")
logger.info(f"   Claims Verified: {stats['claims_verified']}")
logger.info(f"   Color Claims Found: {stats['color_claims']}")

if stats['total_captions_generated'] > 0:
    logger.info(f"   Avg Claims per Caption: {stats['total_claims_extracted']/stats['total_captions_generated']:.2f}")

filter_rate = stats['claims_filtered'] / (stats['total_claims_extracted'] + stats['claims_filtered']) * 100 if (stats['total_claims_extracted'] + stats['claims_filtered']) > 0 else 0
logger.info(f"\nFiltering Impact:")
logger.info(f"   Filter Rate: {filter_rate:.1f}% of noun chunks removed")
logger.info(f"   This reduces false positive 'Object Existence' hallucinations")

logger.info(f"\nHallucination Detection:")
logger.info(f"   Images with Hallucinations: {stats['images_with_hallucinations']}/{stats['total_processed']}")
logger.info(f"   Total Hallucinations Detected: {stats['total_hallucinations']}")

if stats['total_claims_extracted'] > 0:
    hall_rate = stats['total_hallucinations'] / stats['total_claims_extracted'] * 100
    logger.info(f"   Hallucination Rate: {hall_rate:.1f}%")

if error_images:
    logger.info(f"\nErrors Encountered:")
    for err in error_images[:5]:
        logger.info(f"   {err['image_id']}: {err['error']}")
    if len(error_images) > 5:
        logger.info(f"   ... and {len(error_images) - 5} more")

logger.info(f"\n" + "="*80)
logger.info("                    NEXT STEPS")
logger.info("="*80)
logger.info("\n1. Calculate metrics with confidence threshold:")
logger.info(f"   python improved_calc_metrics.py --confidence_threshold 0.7")
logger.info("\n2. Start human annotation (run locally):")
logger.info(f"   streamlit run improved_game_ui.py")
logger.info("\n3. After annotation, create gold dataset:")
logger.info(f"   python create_gold_dataset.py")
logger.info("\n4. Train model on gold dataset:")
logger.info(f"   python train_qwen.py")
logger.info("="*80)

logger.info(f"\nLog saved to: {log_filename}")
logger.info(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("="*80)