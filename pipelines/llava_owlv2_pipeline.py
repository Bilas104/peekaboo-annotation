"""
LLaVA + OWLv2 Hallucination Detection Pipeline
-----------------------------------------------
Uses:
- LLaVA-1.5-7B for captioning
- OWLv2 (improved) for object detection/verification

OWLv2 improvements over OWL-ViT:
- Better detection accuracy
- Improved small object detection
- More robust to varied prompts
"""

import torch
import spacy
import os
import json
import argparse
import logging
import sys
from datetime import datetime
from PIL import Image
from transformers import (
    AutoProcessor, 
    LlavaForConditionalGeneration, 
    BitsAndBytesConfig,
    Owlv2Processor,
    Owlv2ForObjectDetection
)
import numpy as np

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/llava_owlv2_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
parser.add_argument("--output_file", type=str, default="data/output/results_llava_owlv2.json")
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--confidence_threshold", type=float, default=0.1)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("="*70)
logger.info("  LLaVA + OWLv2 HALLUCINATION DETECTION PIPELINE")
logger.info("="*70)
logger.info(f"Device: {DEVICE}")
logger.info(f"Images: {args.image_dir}")
logger.info(f"Limit: {args.limit}")
logger.info(f"Output: {args.output_file}")
logger.info(f"Confidence Threshold: {args.confidence_threshold}")
logger.info("="*70)

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

# 1. Load LLaVA (Captioner)
logger.info("\nLoading LLaVA-1.5-7B (Captioner)...")
llava_model_id = "llava-hf/llava-1.5-7b-hf"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

try:
    llava_processor = AutoProcessor.from_pretrained(llava_model_id)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        llava_model_id, 
        quantization_config=quantization_config, 
        device_map="auto"
    )
    logger.info("LLaVA loaded successfully")
except Exception as e:
    logger.error(f"Failed to load LLaVA: {e}")
    logger.error("Ensure you have installed: pip install accelerate bitsandbytes")
    exit(1)

# 2. Load OWLv2 (Verifier) - UPGRADED FROM OWL-ViT
logger.info("\nLoading OWLv2 (Verifier)...")
owlv2_model_id = "google/owlv2-base-patch16-ensemble"

try:
    owl_processor = Owlv2Processor.from_pretrained(owlv2_model_id)
    owl_model = Owlv2ForObjectDetection.from_pretrained(owlv2_model_id).to(DEVICE)
    owl_model.eval()
    logger.info("OWLv2 loaded successfully")
except Exception as e:
    logger.error(f"Failed to load OWLv2: {e}")
    logger.error("Trying alternative OWLv2 model...")
    try:
        # Fallback to smaller OWLv2 model
        owlv2_model_id = "google/owlv2-base-patch16"
        owl_processor = Owlv2Processor.from_pretrained(owlv2_model_id)
        owl_model = Owlv2ForObjectDetection.from_pretrained(owlv2_model_id).to(DEVICE)
        owl_model.eval()
        logger.info(f"OWLv2 (fallback) loaded successfully: {owlv2_model_id}")
    except Exception as e2:
        logger.error(f"Failed to load OWLv2 fallback: {e2}")
        exit(1)

logger.info("\nAll models loaded. Starting processing...")

# --- HELPER FUNCTIONS ---

def get_caption_llava(image):
    """
    Generates a caption using LLaVA.
    """
    prompt = "USER: <image>\nWrite a short, descriptive caption for this image.\nASSISTANT:"
    
    inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        generate_ids = llava_model.generate(
            **inputs, 
            max_new_tokens=60,
            do_sample=True,
            temperature=0.2,
            top_p=0.9
        )
    
    full_output = llava_processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    if "ASSISTANT:" in full_output:
        caption = full_output.split("ASSISTANT:")[-1].strip()
    else:
        caption = full_output.strip()
        
    return caption

def extract_claims(caption):
    """
    Extract claims from caption using spaCy NLP.
    """
    doc = nlp(caption)
    claims = []
    
    SKIP_TERMS = {
        'field', 'road', 'area', 'scene', 'background', 'ground', 'place', 'spot',
        'way', 'side', 'end', 'top', 'bottom', 'front', 'back', 'center',
        'group', 'bunch', 'collection', 'set', 'pair', 'lot',
        'thing', 'something', 'part', 'piece', 'section', 'portion',
        'it', 'its', 'them', 'their', 'one', 'ones',
        'hay', 'grass', 'dirt', 'sand', 'water', 'snow', 'ice',
        'sunlight', 'shadow', 'light', 'darkness', 'shade', 'image', 'picture', 'photo'
    }
    
    BODY_PARTS = {
        'mouth', 'eye', 'nose', 'ear', 'hand', 'leg', 'arm', 'foot',
        'head', 'tail', 'wing', 'paw', 'face'
    }
    
    number_words = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "a": 1, "an": 1, "several": 3, "few": 2, "couple": 2,
        "many": 5, "some": 2
    }
    
    colors = ["red", "blue", "green", "yellow", "black", "white", 
              "orange", "purple", "brown", "pink", "gray", "grey", 
              "silver", "gold", "beige", "tan"]
    
    for chunk in doc.noun_chunks:
        text = chunk.text.lower()
        root = chunk.root.text.lower()
        
        if root in SKIP_TERMS: continue
        if chunk.root.pos_ == "PRON": continue
        if root in BODY_PARTS and len(chunk) < 3: continue
        if len(chunk.text.split()) == 1 and root in ['a', 'an', 'the']: continue
        
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
    
    return claims

def verify_single_object_owlv2(image, object_name, threshold=0.1):
    """
    Verify object existence using OWLv2.
    OWLv2 uses slightly different API than OWL-ViT.
    """
    # OWLv2 expects queries as list of lists
    inputs = owl_processor(text=[[object_name]], images=image, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = owl_model(**inputs)
    
    # Post-process results
    target_sizes = torch.Tensor([image.size[::-1]]).to(DEVICE)
    results = owl_processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=target_sizes
    )[0]
    
    boxes = results["boxes"].cpu().numpy() if len(results["boxes"]) > 0 else np.array([])
    scores = results["scores"].cpu().numpy() if len(results["scores"]) > 0 else np.array([])
    
    return boxes, scores

def extract_color_from_region(image, box):
    """
    Extract dominant color from a bounding box region.
    """
    x1, y1, x2, y2 = [int(coord) for coord in box]
    
    # Guard against invalid boxes
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    
    if x2 <= x1 or y2 <= y1:
        return "unknown"
    
    region = image.crop((x1, y1, x2, y2))
    region_array = np.array(region)
    
    if region_array.size == 0:
        return "unknown"
    
    # Handle grayscale images
    if len(region_array.shape) == 2:
        avg = region_array.mean()
        if avg > 200: return "white"
        if avg < 60: return "black"
        return "gray"
    
    avg_color = region_array.mean(axis=(0, 1))
    
    if len(avg_color) < 3:
        return "unknown"
    
    r, g, b = avg_color[:3]
    
    # Color classification
    if r > 200 and g > 200 and b > 200: return "white"
    if r < 60 and g < 60 and b < 60: return "black"
    if r > 150 and r > g * 1.5 and r > b * 1.5: return "red"
    if g > 150 and g > r * 1.3 and g > b * 1.3: return "green"
    if b > 150 and b > r * 1.3 and b > g * 1.3: return "blue"
    if r > 180 and g > 150 and b < 100: return "yellow"
    if r > 180 and g < 150 and g > 100 and b < 100: return "orange"
    if r > 150 and b > 150 and g < 100: return "purple"
    if r > 150 and g > 100 and b > 100 and r > g and r > b: return "pink"
    if r > 100 and g > 80 and b < 80 and r > g: return "brown"
    if abs(r - g) < 30 and abs(g - b) < 30 and r > 100 and r < 200: return "gray"
    
    return "unknown"

def verify_claims_owlv2(image, claims, threshold=0.1):
    """
    Verify all claims using OWLv2.
    """
    results = []
    if not claims:
        return []
    
    for claim in claims:
        object_name = claim['phrase']
        expected_count = claim['exp_count']
        expected_color = claim['exp_color']
        has_explicit_count = claim['has_explicit_count']
        
        boxes, scores = verify_single_object_owlv2(image, object_name, threshold)
        detected_count = len(boxes)
        
        h_type = "None"
        h_confidence = 0.0
        color_match = None
        detected_colors = []
        
        # 1. Existence Check
        if detected_count == 0:
            h_type = "Object Existence"
            h_confidence = 0.9  # High confidence for non-detection
        
        # 2. Count Check (only for explicit counts > 1)
        elif has_explicit_count and expected_count > 1:
            count_diff = abs(detected_count - expected_count)
            if count_diff > 1:  # Tolerance of 1
                h_type = "Count"
                h_confidence = min(0.85, count_diff / expected_count)
        
        # 3. Color Check
        if detected_count > 0 and expected_color:
            for box in boxes:
                c = extract_color_from_region(image, box)
                detected_colors.append(c)
            
            if expected_color in detected_colors:
                color_match = True
            else:
                color_match = False
                if h_type == "None":
                    h_type = "Attribute (Color)"
                    h_confidence = 0.8
        
        results.append({
            "claim": claim['full_text'],
            "hallucination_type": h_type,
            "confidence": round(h_confidence, 2),
            "detected_count": detected_count,
            "expected_count": expected_count,
            "expected_color": expected_color,
            "detected_colors": detected_colors,
            "color_match": color_match,
            "detection_scores": [round(float(s), 3) for s in scores[:5]] if len(scores) > 0 else []
        })
    
    return results

# --- MAIN LOOP ---
if not os.path.exists(args.image_dir):
    logger.error(f"Image directory '{args.image_dir}' does not exist.")
    exit(1)

images_list = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
logger.info(f"\nFound {len(images_list)} images. Processing {min(len(images_list), args.limit)}...")

stats = {
    "total_processed": 0,
    "successful": 0,
    "with_hallucinations": 0,
    "errors": 0,
    "hallucination_types": {"Object Existence": 0, "Count": 0, "Attribute (Color)": 0}
}

data = []

for i, img_name in enumerate(images_list[:args.limit]):
    if i % 10 == 0:
        logger.info(f"Processing {i}/{min(len(images_list), args.limit)}...")
        
    try:
        path = os.path.join(args.image_dir, img_name)
        image = Image.open(path).convert("RGB")
        
        # 1. Get Caption (LLaVA)
        caption = get_caption_llava(image)
        
        # 2. Extract Claims
        claims = extract_claims(caption)
        
        # 3. Verify with OWLv2
        verification = verify_claims_owlv2(image, claims, threshold=args.confidence_threshold)
        
        # Track statistics
        stats["total_processed"] += 1
        stats["successful"] += 1
        
        has_hallucination = False
        for v in verification:
            if v['hallucination_type'] != "None":
                has_hallucination = True
                stats["hallucination_types"][v['hallucination_type']] = \
                    stats["hallucination_types"].get(v['hallucination_type'], 0) + 1
        
        if has_hallucination:
            stats["with_hallucinations"] += 1
            
        data.append({
            "image_id": img_name,
            "caption": caption,
            "model": "llava-1.5-7b",
            "detector": "owlv2-base-patch16-ensemble",
            "verification": verification
        })
        
    except Exception as e:
        logger.error(f"Error on {img_name}: {e}")
        stats["errors"] += 1

# Save results
with open(args.output_file, "w") as f:
    json.dump(data, f, indent=2)

# --- SUMMARY ---
logger.info("\n" + "="*70)
logger.info("  PIPELINE COMPLETE - SUMMARY")
logger.info("="*70)
logger.info(f"\n  Total Images:              {stats['total_processed']}")
logger.info(f"  Successful:                {stats['successful']}")
logger.info(f"  Errors:                    {stats['errors']}")
logger.info(f"  With Hallucinations:       {stats['with_hallucinations']} ({stats['with_hallucinations']/stats['successful']*100:.1f}%)")

logger.info(f"\n  Hallucination Breakdown:")
for h_type, count in stats["hallucination_types"].items():
    logger.info(f"    {h_type:25s}  {count}")

logger.info(f"\n  Output saved to: {args.output_file}")
logger.info(f"  Log saved to: {log_filename}")
logger.info("="*70)