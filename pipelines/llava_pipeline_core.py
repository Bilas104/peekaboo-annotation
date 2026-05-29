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
    OwlViTProcessor, 
    OwlViTForObjectDetection
)
import numpy as np

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/llava_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
parser.add_argument("--output_file", type=str, default="data/output/results_llava.json")
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--confidence_threshold", type=float, default=0.12) # Slightly higher for LLaVA context
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Starting Pipeline - LLaVA Edition")
logger.info(f"Device: {DEVICE}")
logger.info(f"Images: {args.image_dir} | Limit: {args.limit}")

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
logger.info("Loading LLaVA-1.5-7B (Captioner)...")
llava_model_id = "llava-hf/llava-1.5-7b-hf"

# Configure 4-bit quantization to save memory
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
except Exception as e:
    logger.error(f"Failed to load LLaVA: {e}")
    logger.error("Ensure you have installed: pip install accelerate bitsandbytes")
    exit(1)

# 2. Load OWL-ViT (Verifier)
logger.info("Loading OWL-ViT (Verifier)...")
owl_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
# OWL-ViT is small enough to run in full precision or float16 without quantization usually
owl_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(DEVICE)
owl_model.eval()

# --- HELPER FUNCTIONS ---

def get_caption_llava(image):
    """
    Generates a caption using LLaVA.
    LLaVA requires a specific prompt template: USER: <image>\n<prompt>\nASSISTANT:
    """
    # Prompt engineering to get a concise caption similar to COCO style
    prompt = "USER: <image>\nWrite a short, descriptive caption for this image.\nASSISTANT:"
    
    inputs = llava_processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    
    # Generate
    with torch.no_grad():
        generate_ids = llava_model.generate(
            **inputs, 
            max_new_tokens=60,
            do_sample=True,
            temperature=0.2, # Low temperature for more factual captions
            top_p=0.9
        )
    
    # Decode and clean
    full_output = llava_processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
    
    # Extract only the assistant's response
    # Output usually looks like: "USER: ... ASSISTANT: A cat sitting on a mat."
    if "ASSISTANT:" in full_output:
        caption = full_output.split("ASSISTANT:")[-1].strip()
    else:
        caption = full_output.strip()
        
    return caption

def extract_claims(caption):
    """
    Same claim extraction logic as core pipeline to ensure consistency.
    """
    doc = nlp(caption)
    claims = []
    
    # Filter generic/abstract terms that cause false positives
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

def verify_single_object(image, object_name, threshold=0.1):
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
    x1, y1, x2, y2 = [int(coord) for coord in box]
    # Guard against invalid boxes
    if x2 <= x1 or y2 <= y1: return "unknown"
    
    region = image.crop((x1, y1, x2, y2))
    region_array = np.array(region)
    if region_array.size == 0: return "unknown"
    
    avg_color = region_array.mean(axis=(0, 1))
    r, g, b = avg_color
    
    # Simplified color logic (same as original for consistency)
    if r > 200 and g > 200 and b > 200: return "white"
    if r < 60 and g < 60 and b < 60: return "black"
    if r > 150 and r > g*1.5 and r > b*1.5: return "red"
    if g > 150 and g > r*1.3 and g > b*1.3: return "green"
    if b > 150 and b > r*1.3 and b > g*1.3: return "blue"
    
    return "unknown" # Fallback

def verify_claims_improved(image, claims, threshold=0.1):
    results = []
    if not claims: return []
    
    for claim in claims:
        object_name = claim['phrase']
        expected_count = claim['exp_count']
        expected_color = claim['exp_color']
        has_explicit_count = claim['has_explicit_count']
        
        boxes, scores = verify_single_object(image, object_name, threshold)
        detected_count = len(boxes)
        
        h_type = "None"
        h_confidence = 0.0
        color_match = None
        detected_colors = []
        
        # 1. Existence Check
        if detected_count == 0:
            h_type = "Object Existence"
            h_confidence = 1.0
        
        # 2. Count Check
        elif has_explicit_count and expected_count > 1:
            if abs(detected_count - expected_count) > 1: # Tolerance
                h_type = "Count"
                h_confidence = min(1.0, abs(detected_count - expected_count) / expected_count)
        
        # 3. Color Check
        if detected_count > 0 and expected_color:
            for box in boxes:
                c = extract_color_from_region(image, box)
                detected_colors.append(c)
            
            # Simple check: did we see the expected color at least once?
            # (Refining this to specific object mapping is complex, using loose matching here)
            if expected_color in detected_colors:
                color_match = True
            else:
                # If we detected colors but none matched
                color_match = False
                if h_type == "None":
                    h_type = "Attribute (Color)"
                    h_confidence = 0.7
        
        results.append({
            "claim": claim['full_text'],
            "hallucination_type": h_type,
            "confidence": round(h_confidence, 2),
            "detected_count": detected_count,
            "expected_count": expected_count,
            "expected_color": expected_color,
            "detected_colors": detected_colors
        })
    
    return results

# --- MAIN LOOP ---
if not os.path.exists(args.image_dir):
    logger.error(f"Image directory '{args.image_dir}' does not exist.")
    exit(1)

images_list = [f for f in os.listdir(args.image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
logger.info(f"Found {len(images_list)} images. Processing {min(len(images_list), args.limit)}...")

stats = {"captions": 0, "hallucinations": 0}
data = []

for i, img_name in enumerate(images_list[:args.limit]):
    if i % 5 == 0:
        logger.info(f"Processing {i}...")
        
    try:
        path = os.path.join(args.image_dir, img_name)
        image = Image.open(path).convert("RGB")
        
        # 1. Get Caption (LLaVA)
        cap = get_caption_llava(image)
        stats["captions"] += 1
        
        # 2. Extract Claims
        claims = extract_claims(cap)
        
        # 3. Verify
        verification = verify_claims_improved(image, claims, threshold=args.confidence_threshold)
        
        # Check if hallucination exists
        if any(v['hallucination_type'] != "None" for v in verification):
            stats["hallucinations"] += 1
            
        data.append({
            "image_id": img_name,
            "caption": cap,
            "model": "llava-1.5-7b",
            "verification": verification
        })
        
    except Exception as e:
        logger.error(f"Error on {img_name}: {e}")

# Save
with open(args.output_file, "w") as f:
    json.dump(data, f, indent=2)

logger.info(f"Pipeline Complete.")
logger.info(f"Images: {stats['captions']}")
logger.info(f"Images with Hallucinations detected: {stats['hallucinations']}")
logger.info(f"Saved to {args.output_file}")