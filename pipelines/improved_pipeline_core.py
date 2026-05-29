import torch
import spacy
import os
import json
import argparse
import logging
import sys
from datetime import datetime
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import numpy as np

# --- LOGGING SETUP ---
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

# --- CONFIG ---
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, default="data/images")
parser.add_argument("--output_file", type=str, default="data/output/results.json")
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--confidence_threshold", type=float, default=0.08)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Starting Pipeline - OPTION C (Hybrid Approach)")
logger.info(f"Device: {DEVICE}")
logger.info(f"Images: {args.image_dir} | Limit: {args.limit}")
logger.info(f"Detection Threshold: {args.confidence_threshold}")

# Ensure output directory exists
os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

# Load Spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("Spacy model not found. Downloading...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- MODEL LOADING ---
logger.info("Loading BLIP-2 (Captioner)...")
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", 
    torch_dtype=torch.float16, 
    device_map="auto"
)

logger.info("Loading OWL-ViT (Verifier)...")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(DEVICE)
owl_model.eval()

# --- HELPER FUNCTIONS ---
def get_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt").to(blip_model.device, torch.float16)
    generated_ids = blip_model.generate(**inputs, max_new_tokens=50)
    return blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def extract_claims(caption):
    """
    Enhanced claim extraction with generic term filtering (OPTION C)
    Filters out abstract/generic terms that OWL-ViT cannot detect
    """
    doc = nlp(caption)
    claims = []
    
    # OPTION C: Filter generic/abstract terms that cause false positives
    SKIP_TERMS = {
        # Abstract locations
        'field', 'road', 'area', 'scene', 'background', 'ground', 'place', 'spot',
        'way', 'side', 'end', 'top', 'bottom', 'front', 'back', 'center',
        
        # Generic groupings
        'group', 'bunch', 'collection', 'set', 'pair', 'lot',
        
        # Abstract concepts
        'thing', 'something', 'part', 'piece', 'section', 'portion',
        
        # Pronouns and vague references
        'it', 'its', 'them', 'their', 'one', 'ones',
        
        # Textures/materials (hard to detect)
        'hay', 'grass', 'dirt', 'sand', 'water', 'snow', 'ice',
        
        # Lighting/conditions
        'sunlight', 'shadow', 'light', 'darkness', 'shade'
    }
    
    # Additional filtering: body parts attached to detected objects
    BODY_PARTS = {
        'mouth', 'eye', 'nose', 'ear', 'hand', 'leg', 'arm', 'foot',
        'head', 'tail', 'wing', 'paw', 'face'
    }
    
    # Number word mapping
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1, "several": 3, "few": 2, "couple": 2,
        "many": 5, "some": 2
    }
    
    # Color list
    colors = ["red", "blue", "green", "yellow", "black", "white", 
              "orange", "purple", "brown", "pink", "gray", "grey", 
              "silver", "gold", "beige", "tan"]
    
    filtered_count = 0
    
    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        root = chunk.root.text.lower()
        
        # Filter 1: Skip generic terms
        if root in SKIP_TERMS:
            filtered_count += 1
            logger.debug(f"Filtered generic term: '{chunk.text}'")
            continue
        
        # Filter 2: Skip pronouns
        if chunk.root.pos_ == "PRON":
            filtered_count += 1
            logger.debug(f"Filtered pronoun: '{chunk.text}'")
            continue
        
        # Filter 3: Skip body parts unless they're the main subject
        # (e.g., skip "its mouth" but keep "a dog's mouth" if it's prominent)
        if root in BODY_PARTS and len(chunk) < 3:  # Short body part references
            filtered_count += 1
            logger.debug(f"Filtered body part: '{chunk.text}'")
            continue
        
        # Filter 4: Skip very short chunks (usually determiners/articles)
        if len(chunk.text.split()) == 1 and root in ['a', 'an', 'the']:
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
        
        # Extract color
        color = None
        for col in colors:
            if col in text:
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

def verify_single_object(image, object_name, threshold=0.08):
    """Verify a single object with individual processing for better accuracy"""
    inputs = owl_processor(text=[[object_name]], images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = owl_model(**inputs)
    
    target_sizes = torch.Tensor([image.size[::-1]]).to(DEVICE)
    results = owl_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    
    boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.array([])
    scores = results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([])
    
    return boxes, scores

def extract_color_from_region(image, box):
    """
    Extract dominant color from bounding box region
    Returns most common color name from predefined list
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    region = image.crop((x1, y1, x2, y2))
    
    # Get dominant color
    region_array = np.array(region)
    avg_color = region_array.mean(axis=(0, 1))
    
    # Simple color classification
    r, g, b = avg_color
    
    color_ranges = {
        "red": lambda r, g, b: r > 150 and r > g * 1.5 and r > b * 1.5,
        "blue": lambda r, g, b: b > 150 and b > r * 1.3 and b > g * 1.3,
        "green": lambda r, g, b: g > 150 and g > r * 1.3 and g > b * 1.3,
        "yellow": lambda r, g, b: r > 180 and g > 180 and b < 100,
        "orange": lambda r, g, b: r > 200 and g > 100 and g < 200 and b < 100,
        "purple": lambda r, g, b: r > 100 and b > 100 and g < r * 0.7,
        "pink": lambda r, g, b: r > 180 and g < 150 and b > 100 and b < 200,
        "brown": lambda r, g, b: r > 80 and r < 150 and g > 50 and g < 120 and b > 30 and b < 90,
        "white": lambda r, g, b: r > 200 and g > 200 and b > 200,
        "black": lambda r, g, b: r < 60 and g < 60 and b < 60,
        "gray": lambda r, g, b: abs(r - g) < 30 and abs(g - b) < 30 and 60 < r < 200,
    }
    
    for color_name, check_fn in color_ranges.items():
        if check_fn(r, g, b):
            return color_name
    
    return "unknown"

def verify_claims_improved(image, claims, threshold=0.08):
    """
    Improved verification with:
    1. Individual object detection for each claim
    2. Attribute verification (color)
    3. Smarter count matching with confidence levels
    """
    results = []
    if not claims:
        return []
    
    for claim in claims:
        object_name = claim['phrase']
        expected_count = claim['exp_count']
        expected_color = claim['exp_color']
        has_explicit_count = claim['has_explicit_count']
        
        # Detect object
        boxes, scores = verify_single_object(image, object_name, threshold)
        detected_count = len(boxes)
        
        # Initialize result
        h_type = "None"
        h_confidence = 0.0
        color_match = None
        detected_colors = []
        
        # Check 1: Object Existence
        if detected_count == 0:
            h_type = "Object Existence"
            h_confidence = 1.0
        
        # Check 2: Count Verification (only if explicit count mentioned)
        elif has_explicit_count and expected_count > 1:
            # Allow ±1 tolerance for counts
            if abs(detected_count - expected_count) > 1:
                h_type = "Count"
                h_confidence = min(1.0, abs(detected_count - expected_count) / expected_count)
        
        # Check 3: Color Verification (if object exists and color mentioned)
        if detected_count > 0 and expected_color:
            for box in boxes:
                detected_color = extract_color_from_region(image, box)
                detected_colors.append(detected_color)
            
            # Check if any detected box has the expected color
            color_match = expected_color in detected_colors
            
            if not color_match and h_type == "None":
                h_type = "Attribute (Color)"
                h_confidence = 0.8
        
        results.append({
            "claim": claim['full_text'],
            "hallucination_type": h_type,
            "confidence": round(h_confidence, 2),
            "detected_count": detected_count,
            "expected_count": expected_count,
            "detection_scores": [round(float(s), 3) for s in scores] if len(scores) > 0 else [],
            "expected_color": expected_color,
            "detected_colors": detected_colors if expected_color else None,
            "color_match": color_match if expected_color else None
        })
    
    return results

# --- MAIN LOOP (MODIFIED FOR RESUMING) ---
if not os.path.exists(args.image_dir):
    logger.error(f"Image directory '{args.image_dir}' does not exist.")
    exit(1)

# 1. Initialize Stats (THIS WAS MISSING)
stats = {
    "total_captions_generated": 0,
    "total_claims_extracted": 0,
    "claims_filtered": 0,
    "claims_verified": 0
}

# 2. Load existing results if they exist
existing_data = []
processed_ids = set()

if os.path.exists(args.output_file):
    try:
        with open(args.output_file, 'r') as f:
            existing_data = json.load(f)
            processed_ids = {item['image_id'] for item in existing_data}
        logger.info(f"Resuming... Found {len(existing_data)} images already processed.")
    except json.JSONDecodeError:
        logger.warning(f"Could not read {args.output_file}. Starting fresh.")

# 3. Filter images list
all_images = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
new_images = [img for img in all_images if img not in processed_ids]

logger.info(f"Found {len(all_images)} total images.")
logger.info(f"Skipping {len(processed_ids)} already done.")
logger.info(f"Queueing {min(len(new_images), args.limit)} new images...")

# 4. Process only new images
new_data = []
success_count = 0

for i, img_name in enumerate(new_images[:args.limit]):
    if i % 10 == 0:
        logger.info(f"Processing image {i}/{min(len(new_images), args.limit)}...")
    try:
        path = os.path.join(args.image_dir, img_name)
        image = Image.open(path).convert("RGB")
        
        cap = get_caption(image)
        stats["total_captions_generated"] += 1
        
        # Extract claims
        all_chunks = len(list(nlp(cap).noun_chunks))
        claims = extract_claims(cap)
        stats["claims_filtered"] += (all_chunks - len(claims))
        stats["total_claims_extracted"] += len(claims)
        
        verification = verify_claims_improved(image, claims, threshold=args.confidence_threshold)
        stats["claims_verified"] += len(verification)
        
        # Add to new batch
        new_data.append({
            "image_id": img_name,
            "caption": cap,
            "verification": verification
        })
        success_count += 1
        
        # Periodic Save (Safeguard against crashes)
        if success_count % 50 == 0:
            combined_data = existing_data + new_data
            with open(args.output_file, "w") as f:
                json.dump(combined_data, f, indent=2)
            logger.info(f"Checkpoint saved ({len(combined_data)} total images)")

    except Exception as e:
        logger.error(f"Error on {img_name}: {e}")

# 5. Final Save (Append new data to existing)
if new_data:
    final_dataset = existing_data + new_data
    with open(args.output_file, "w") as f:
        json.dump(final_dataset, f, indent=2)
    
    # Log statistics
    logger.info("\n" + "="*80)
    logger.info("                    PIPELINE STATISTICS")
    logger.info("="*80)
    logger.info(f"   New Images Added: {len(new_data)}")
    logger.info(f"   Total Dataset Size: {len(final_dataset)}")
    logger.info(f"   Claims Extracted (New): {stats['total_claims_extracted']}")
    logger.info(f"   Generic Filtered (New): {stats['claims_filtered']}")
else:
    logger.info("\nNo new images to process.")