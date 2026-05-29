import streamlit as st
import json
import os
from PIL import Image
from datetime import datetime
import logging

st.set_page_config(layout="wide", page_title="Peekaboo Annotator", page_icon="🔍")

# --- LOGGING SETUP ---
os.makedirs("logs", exist_ok=True)
log_filename = f"logs/annotation_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIG ---
AI_RESULTS_FILE = "data/output/results.json"
HUMAN_OUTPUT_FILE = "data/output/human_annotations.json"
IMG_DIR = "data/images"

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .claim-box {
        background-color: #ffffff;
        padding: 1rem;
        border-left: 4px solid #1E88E5;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .hallucination-badge {
        background-color: #ff4444;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    .accurate-badge {
        background-color: #00C851;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    .corrected-caption-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #2196F3;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD AI DATA ---
if not os.path.exists(AI_RESULTS_FILE):
    st.error(f"Results file not found: {AI_RESULTS_FILE}")
    st.info("Please run the pipeline first.")
    st.stop()

with open(AI_RESULTS_FILE) as f:
    ai_data = json.load(f)

# --- INITIALIZE SESSION STATE ---
if 'idx' not in st.session_state:
    st.session_state.idx = 0
    logger.info(f"New annotation session initialized")

if 'annotations_made' not in st.session_state:
    st.session_state.annotations_made = 0

if 'active_filters' not in st.session_state:
    st.session_state.active_filters = []

# --- HELPER FUNCTIONS ---

def get_annotated_ids():
    """Return a set of image_ids that have already been annotated"""
    if os.path.exists(HUMAN_OUTPUT_FILE):
        try:
            with open(HUMAN_OUTPUT_FILE, 'r') as f:
                data = json.load(f)
                return {item['image_id'] for item in data}
        except:
            return set()
    return set()

def save_annotation(idx, annotations, notes="", corrected_caption=""):
    """Save human annotation with logging"""
    if os.path.exists(HUMAN_OUTPUT_FILE):
        with open(HUMAN_OUTPUT_FILE, 'r') as f:
            try: database = json.load(f)
            except: database = []
    else:
        database = []

    entry = {
        "image_id": ai_data[idx]['image_id'],
        "caption": ai_data[idx]['caption'],
        "human_corrected_caption": corrected_caption if corrected_caption else ai_data[idx]['caption'],
        "human_labels": annotations,
        "notes": notes,
        "timestamp": datetime.now().isoformat(),
        "annotator_session": log_filename
    }

    # Upsert logic
    database = [d for d in database if d['image_id'] != entry['image_id']]
    database.append(entry)

    with open(HUMAN_OUTPUT_FILE, 'w') as f:
        json.dump(database, f, indent=2)
    
    correction_status = "with correction" if corrected_caption else "no correction"
    logger.info(f"Saved {entry['image_id']} ({correction_status})")
    st.session_state.annotations_made += 1

def filter_images_multi(data, filters):
    """
    Advanced filtering supporting multiple criteria (AND logic).
    """
    # If no filters, return everything
    if not filters:
        return list(range(len(data)))

    annotated_ids = get_annotated_ids()
    indices = []

    for i, item in enumerate(data):
        conditions = []
        
        # 1. Unannotated Only
        if "Unannotated Only" in filters:
            conditions.append(item['image_id'] not in annotated_ids)
            
        # 2. Annotated Only (Review Mode)
        if "Annotated Only" in filters:
            conditions.append(item['image_id'] in annotated_ids)

        # 3. Has Hallucinations (Any confidence)
        if "Has Hallucinations" in filters:
            has_hall = any(v['hallucination_type'] != "None" for v in item['verification'])
            conditions.append(has_hall)
            
        # 4. High Confidence Only (> 0.8)
        if "High Confidence Halls" in filters:
            high_conf = any(v['hallucination_type'] != "None" and v.get('confidence', 0) >= 0.8 
                            for v in item['verification'])
            conditions.append(high_conf)
            
        # 5. Accurate Only
        if "Accurate Only" in filters:
            is_accurate = all(v['hallucination_type'] == "None" for v in item['verification'])
            conditions.append(is_accurate)

        # If all selected conditions match, keep the index
        if all(conditions):
            indices.append(i)
            
    return indices

def reset_index():
    """Callback to reset index when filters change"""
    st.session_state.idx = 0

def update_index_from_input():
    """Callback to jump to index"""
    # Convert from 1-based (UI) to 0-based (Code)
    new_val = st.session_state.jump_input - 1
    if 0 <= new_val < len(filtered_indices):
        st.session_state.idx = new_val

# --- SIDEBAR ---
st.sidebar.markdown("## Peekaboo Annotator")
st.sidebar.markdown("---")

# Stats
annotated_ids = get_annotated_ids()
annotated_count = len(annotated_ids)
st.sidebar.metric("Total Progress", f"{annotated_count}/{len(ai_data)}")
st.sidebar.metric("Session Count", st.session_state.annotations_made)
st.sidebar.progress(annotated_count / len(ai_data) if ai_data else 0)

st.sidebar.markdown("---")

# --- MULTI FILTER ---
st.sidebar.markdown("### 🔍 Filters")
filter_options = [
    "Unannotated Only", 
    "Annotated Only",
    "Has Hallucinations", 
    "High Confidence Halls", 
    "Accurate Only"
]

selected_filters = st.sidebar.multiselect(
    "Combine Filters:",
    options=filter_options,
    default=[],
    key="active_filters",
    on_change=reset_index,
    help="Select multiple to narrow down (e.g., 'Unannotated Only' + 'Has Hallucinations')"
)

# Apply Filter
filtered_indices = filter_images_multi(ai_data, selected_filters)

if not filtered_indices:
    st.sidebar.error("No images match these filters.")
    st.stop()

st.sidebar.info(f"Showing **{len(filtered_indices)}** images")

# --- INDEX JUMP ---
st.sidebar.markdown("### ⏭️ Navigation")

# Create a number input that syncs with session state
# We use a distinct key 'jump_input' and a callback to update 'idx'
current_display_idx = st.session_state.idx + 1
st.sidebar.number_input(
    "Jump to Image #",
    min_value=1,
    max_value=len(filtered_indices),
    value=current_display_idx,
    step=1,
    key="jump_input",
    on_change=update_index_from_input
)

if st.sidebar.button("Random Image"):
    import random
    st.session_state.idx = random.randint(0, len(filtered_indices) - 1)
    st.rerun()

st.sidebar.markdown("---")

# --- DOWNLOAD ---
if st.sidebar.button("📥 Download Annotations"):
    if os.path.exists(HUMAN_OUTPUT_FILE):
        with open(HUMAN_OUTPUT_FILE) as f:
            data_str = f.read()
        st.sidebar.download_button(
            "Save JSON",
            data_str,
            file_name=f"human_annotations_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# --- MAIN UI ---
st.markdown('<div class="main-title">Peekaboo: Spot the Hallucination</div>', unsafe_allow_html=True)

# Nav Buttons (Top)
col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 3, 1])

with col_nav1:
    if st.button("⬅️ Previous", use_container_width=True):
        if st.session_state.idx > 0:
            st.session_state.idx -= 1
            st.rerun()

with col_nav2:
    if st.button("Next ➡️", use_container_width=True):
        if st.session_state.idx < len(filtered_indices) - 1:
            st.session_state.idx += 1
            st.rerun()

with col_nav4:
    if st.button("Skip ⏩", use_container_width=True):
        st.session_state.idx = min(len(filtered_indices) - 1, st.session_state.idx + 1)
        st.rerun()

# --- CONTENT DISPLAY ---
current_idx = st.session_state.idx
actual_data_idx = filtered_indices[current_idx]
item = ai_data[actual_data_idx]

# Check if already annotated (for visual cue)
is_annotated = item['image_id'] in annotated_ids
status_emoji = "✅ Annotated" if is_annotated else "⏳ Pending"

st.caption(f"Image {current_idx + 1} of {len(filtered_indices)} (Filtered) | ID: {item['image_id']} | Status: {status_emoji}")
st.progress((current_idx + 1) / len(filtered_indices))

col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown("### Image")
    path = os.path.join(IMG_DIR, item['image_id'])
    if os.path.exists(path):
        image = Image.open(path)
        st.image(image, use_container_width=True)
    else:
        st.error(f"Image file missing: {item['image_id']}")
    
    st.markdown("### Generated Caption")
    st.markdown(f"#### *\"{item['caption']}\"*")
    
    # Quick Summary
    hall_count = sum(1 for v in item['verification'] if v['hallucination_type'] != "None")
    if hall_count > 0:
        st.warning(f"⚠️ AI detected {hall_count} potential issues.")
    else:
        st.success("✨ AI thinks this caption is accurate.")

with col2:
    st.markdown("### Verification & Annotation")
    
    with st.form(key=f"form_{item['image_id']}"):
        human_verdicts = []
        
        # Claims Loop
        for idx, v in enumerate(item['verification']):
            st.markdown('<div class="claim-box">', unsafe_allow_html=True)
            col_claim, col_badge = st.columns([3, 1])
            
            with col_claim:
                st.markdown(f"**Claim {idx+1}:** `{v['claim']}`")
            
            with col_badge:
                if v['hallucination_type'] != "None":
                    st.markdown(f'<span class="hallucination-badge">{v["hallucination_type"]}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="accurate-badge">Accurate</span>', unsafe_allow_html=True)

            # Details expander
            with st.expander("🔍 Show Evidence"):
                st.write(f"Detected: {v['detected_count']} | Expected: {v.get('expected_count', 1)}")
                st.write(f"Confidence: {v.get('confidence', 0):.2f}")
                if v.get('expected_color'):
                    st.write(f"Color: {v.get('detected_colors')} (Expected: {v['expected_color']})")

            # Radio Input
            # Try to pre-fill if this image was already annotated
            default_index = 0 if v['hallucination_type'] == "None" else 1
            
            # Logic: If we are in 'Annotated Only' mode, we might want to load previous answers.
            # But for speed, usually we just stick to AI defaults unless specified.
            
            choice = st.radio(
                "Verdict:", ["Accurate", "Hallucination"],
                key=f"rad_{item['image_id']}_{idx}",
                horizontal=True,
                index=default_index
            )
            
            human_verdicts.append({
                "claim": v['claim'],
                "label": choice,
                "ai_prediction": v['hallucination_type']
            })
            st.markdown('</div>', unsafe_allow_html=True)

        st.divider()
        
        # Correction Section
        st.markdown("#### Corrected Caption")
        
        # Attempt to load existing correction if available
        prev_correction = ""
        if is_annotated and os.path.exists(HUMAN_OUTPUT_FILE):
             with open(HUMAN_OUTPUT_FILE) as f:
                temp_db = json.load(f)
                for entry in temp_db:
                    if entry['image_id'] == item['image_id']:
                        prev_correction = entry.get('human_corrected_caption', '')
                        if prev_correction == item['caption']: prev_correction = ""
                        break

        new_caption = st.text_area(
            "Rewrite if inaccurate:",
            value=prev_correction,
            placeholder="Write what you actually see...",
            help="Leave blank if the original caption is fine."
        )
        
        notes = st.text_area("Notes:", height=70)
        
        # Submit
        col_sub1, col_sub2 = st.columns([3, 1])
        with col_sub2:
            submitted = st.form_submit_button("💾 Save & Next", use_container_width=True, type="primary")
            
        if submitted:
            save_annotation(actual_data_idx, human_verdicts, notes, new_caption.strip())
            st.success("Saved!")
            
            # Logic to auto-advance
            if st.session_state.idx < len(filtered_indices) - 1:
                st.session_state.idx += 1
                st.rerun()
            else:
                st.balloons()
                st.info("End of filtered list.")

# Footer
st.markdown("---")
st.caption("Peekaboo Annotation Tool v2.0")