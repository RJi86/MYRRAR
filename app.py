import os
import urllib.request

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

st.set_page_config(page_title="MYRRAR PreAlpha", layout="centered")
st.title("MYRRAR: Skin Tone Analyzer (PreAlpha)")

# --- Landmark indices for safe skin regions ---
LEFT_CHEEK = [234, 93, 132, 58, 172, 214, 216, 203, 100, 229, 234]
RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 436, 423, 371, 349, 449, 340]
FOREHEAD = [9, 107, 69, 104, 103, 67, 109, 10, 338, 297, 332, 333, 299, 336, 9]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

DEBUG_SHOW_LANDMARKS = False  # show 468 points + polygons + bbox (hidden by default)

@st.cache_resource(show_spinner=True)
def load_face_landmarker():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def apply_clahe_lab(image_bgr, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE in LAB color space.
    Only normalizes Lightness, preserving color information (A and B channels).
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)


def create_region_mask(image_shape, landmarks, region_indices):
    """Create a binary mask for a polygonal region using fillPoly."""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) 
                    for i in region_indices], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    
    return mask


def estimate_skin_tone(l_channel_mean):
    if l_channel_mean > 75:
        return "Light"
    if l_channel_mean > 60:
        return "Medium"
    if l_channel_mean > 45:
        return "Tan"
    return "Deep"


def bbox_from_landmarks(landmarks, w, h, pad=0.02):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x_min = max(0, int((min(xs) - pad) * w))
    x_max = min(w, int((max(xs) + pad) * w))
    y_min = max(0, int((min(ys) - pad) * h))
    y_max = min(h, int((max(ys) + pad) * h))
    return x_min, y_min, x_max, y_max


def draw_polygons(img, landmarks, regions, color=(0, 255, 255), thickness=1):
    """Draw polygons for the specified regions on the image."""
    h, w, _ = img.shape
    out = img.copy()
    for region in regions:
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in region], np.int32)
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    return out


def analyze_skin_colors_before_after(image_bgr, landmarks):
    """
    Analyze skin color across multiple face regions,
    returning both original and CLAHE-normalized values.
    """
    regions = {
        'left_cheek': LEFT_CHEEK,
        'right_cheek': RIGHT_CHEEK,
        'forehead': FOREHEAD,
    }
    
    # Process CLAHE version
    processed = apply_clahe_lab(image_bgr)
    
    results = {}
    
    for region_name, indices in regions.items():
        mask = create_region_mask(image_bgr.shape, landmarks, indices)
        
        # Before (original)
        mean_bgr_before = cv2.mean(image_bgr, mask=mask)[:3]
        
        # After (CLAHE normalized)
        mean_bgr_after = cv2.mean(processed, mask=mask)[:3]
        
        results[region_name] = {
            'before': {
                'BGR': tuple(map(int, mean_bgr_before)),
                'RGB': tuple(map(int, (mean_bgr_before[2], mean_bgr_before[1], mean_bgr_before[0]))),
                'hex': '#{:02x}{:02x}{:02x}'.format(int(mean_bgr_before[2]), int(mean_bgr_before[1]), int(mean_bgr_before[0])),
            },
            'after': {
                'BGR': tuple(map(int, mean_bgr_after)),
                'RGB': tuple(map(int, (mean_bgr_after[2], mean_bgr_after[1], mean_bgr_after[0]))),
                'hex': '#{:02x}{:02x}{:02x}'.format(int(mean_bgr_after[2]), int(mean_bgr_after[1]), int(mean_bgr_after[0])),
            }
        }
    
    # Calculate FINAL average
    avg_before_bgr = tuple(
        int(sum(results[r]['before']['BGR'][c] for r in regions) / len(regions))
        for c in range(3)
    )
    avg_after_bgr = tuple(
        int(sum(results[r]['after']['BGR'][c] for r in regions) / len(regions))
        for c in range(3)
    )
    
    results['final'] = {
        'before': {
            'BGR': avg_before_bgr,
            'RGB': (avg_before_bgr[2], avg_before_bgr[1], avg_before_bgr[0]),
            'hex': '#{:02x}{:02x}{:02x}'.format(avg_before_bgr[2], avg_before_bgr[1], avg_before_bgr[0]),
        },
        'after': {
            'BGR': avg_after_bgr,
            'RGB': (avg_after_bgr[2], avg_after_bgr[1], avg_after_bgr[0]),
            'hex': '#{:02x}{:02x}{:02x}'.format(avg_after_bgr[2], avg_after_bgr[1], avg_after_bgr[0]),
        }
    }
    
    return results, processed


def visualize_skin_analysis_before_after(image_bgr, landmarks):
    """
    Create a visualization showing before/after skin analysis with color swatches.
    Larger swatch panel and text for better readability.
    """
    results, processed = analyze_skin_colors_before_after(image_bgr, landmarks)
    
    h, w, _ = image_bgr.shape
    
    # LARGER swatch panel
    swatch_panel_width = 550
    canvas = np.zeros((h, w * 2 + swatch_panel_width, 3), dtype=np.uint8)
    canvas[:, :, :] = 40  # Dark gray background
    
    # Original with regions outlined
    original_viz = draw_polygons(image_bgr, landmarks, [LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD])
    canvas[:, :w] = original_viz
    
    # Processed with regions outlined
    processed_viz = draw_polygons(processed, landmarks, [LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD])
    canvas[:, w:w*2] = processed_viz
    
    # === Color Swatches Panel ===
    panel_x = w * 2
    swatch_size = 80  # LARGER swatches
    row_height = 130  # More vertical space
    padding = 25
    
    regions = ['left_cheek', 'right_cheek', 'forehead', 'final']
    
    # Title for swatch panel
    cv2.putText(canvas, "BEFORE", (panel_x + padding + 20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "AFTER", (panel_x + padding + swatch_size + 100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    
    for i, region in enumerate(regions):
        y_start = 70 + i * row_height
        
        # Region label
        label = region.replace('_', ' ').title()
        if region == 'final':
            # Make FINAL stand out
            cv2.putText(canvas, "─" * 30, (panel_x + padding, y_start - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
            cv2.putText(canvas, label, (panel_x + padding, y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, label, (panel_x + padding, y_start + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        
        # Before swatch
        before_bgr = results[region]['before']['BGR']
        before_x = panel_x + padding
        before_y = y_start + 35
        cv2.rectangle(canvas,
                      (before_x, before_y),
                      (before_x + swatch_size, before_y + swatch_size),
                      before_bgr, -1)
        cv2.rectangle(canvas,
                      (before_x, before_y),
                      (before_x + swatch_size, before_y + swatch_size),
                      (255, 255, 255), 2)  # White border
        
        # Before hex and RGB labels
        cv2.putText(canvas, results[region]['before']['hex'],
                    (before_x, before_y + swatch_size + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        rgb_text = f"RGB{results[region]['before']['RGB']}"
        cv2.putText(canvas, rgb_text,
                    (before_x, before_y + swatch_size + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
        
        # Arrow between swatches
        arrow_x = before_x + swatch_size + 15
        arrow_y = before_y + swatch_size // 2
        cv2.arrowedLine(canvas, (arrow_x, arrow_y), (arrow_x + 40, arrow_y),
                        (200, 200, 200), 2, tipLength=0.35)
        
        # After swatch
        after_bgr = results[region]['after']['BGR']
        after_x = before_x + swatch_size + 70
        after_y = before_y
        cv2.rectangle(canvas,
                      (after_x, after_y),
                      (after_x + swatch_size, after_y + swatch_size),
                      after_bgr, -1)
        cv2.rectangle(canvas,
                      (after_x, after_y),
                      (after_x + swatch_size, after_y + swatch_size),
                      (255, 255, 255), 2)  # White border
        
        # After hex and RGB labels
        cv2.putText(canvas, results[region]['after']['hex'],
                    (after_x, after_y + swatch_size + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
        rgb_text = f"RGB{results[region]['after']['RGB']}"
        cv2.putText(canvas, rgb_text,
                    (after_x, after_y + swatch_size + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Image labels - LARGER green text
    cv2.putText(canvas, "Original", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "CLAHE Normalized", (w + 15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    return canvas, results


def draw_debug_overlays(image_bgr, result, landmarks,
                        box_color=(0, 0, 255),  # red (BGR)
                        lm_color=(255, 0, 0),   # blue (BGR)
                        poly_color=(0, 255, 255),  # yellow for polygons
                        box_thickness=2,
                        lm_radius=1):
    dbg = image_bgr.copy()
    h, w, _ = dbg.shape

    # Try detector box first
    drew_box = False
    if getattr(result, "face_detections", None):
        for det in result.face_detections:
            rel_box = det.bounding_box
            x_min = int(rel_box.origin_x)
            y_min = int(rel_box.origin_y)
            x_max = x_min + int(rel_box.width)
            y_max = y_min + int(rel_box.height)
            cv2.rectangle(dbg, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
            drew_box = True

    # Fallback: derive bbox from landmarks
    if not drew_box and landmarks:
        x_min, y_min, x_max, y_max = bbox_from_landmarks(landmarks, w, h)
        cv2.rectangle(dbg, (x_min, y_min), (x_max, y_max), box_color, box_thickness)

    # Landmarks
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(dbg, (cx, cy), lm_radius, lm_color, -1, lineType=cv2.LINE_AA)

    # Polygons for our sampling regions
    for region in (LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD):
        pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in region], np.int32)
        cv2.polylines(dbg, [pts], isClosed=True, color=poly_color, thickness=1, lineType=cv2.LINE_AA)

    return dbg


uploaded = st.file_uploader("Upload a clear, front-facing selfie", type=["jpg", "png", "jpeg"])

if uploaded:
    image_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_landmarker = load_face_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = face_landmarker.detect(mp_image)

    if not result.face_landmarks:
        st.error("No face detected. Please upload a clear selfie.")
    else:
        landmarks = result.face_landmarks[0]  # list of 468 NormalizedLandmark

        # Analyze skin colors (CLAHE still used internally for accuracy)
        results, processed = analyze_skin_colors_before_after(image, landmarks)
        
        # === P1: PROMINENT "YOUR SKIN TONE" HERO RESULT ===
        st.markdown("---")
        st.markdown("## 🎨 Your Skin Tone")
        
        # Large hero color swatch for final result
        final_hex = results['final']['after']['hex']
        st.markdown(f'''
        <div style="text-align: center; padding: 20px;">
            <div style="
                width: 200px; 
                height: 200px; 
                background-color: {final_hex}; 
                margin: 0 auto 15px auto; 
                border-radius: 15px;
                border: 4px solid #ddd;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            "></div>
            <div style="font-size: 28px; font-weight: bold; color: #333; margin-bottom: 8px;">
                {final_hex.upper()}
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Estimate skin tone classification (secondary info)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for region in (LEFT_CHEEK, RIGHT_CHEEK, FOREHEAD):
            mask |= create_region_mask(image.shape, landmarks, region)

        normalized_image = apply_clahe_lab(image)
        lab = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        skin_pixels = l_channel[mask == 255]

        if skin_pixels.size >= 500:
            mean_l = float(np.mean(skin_pixels))
            tone = estimate_skin_tone(mean_l)
            st.markdown(f"<div style='text-align: center; font-size: 18px; color: #666;'>Classification: <strong>{tone}</strong></div>", unsafe_allow_html=True)
        
        st.caption("📸 Results may vary with lighting and makeup")
        
        # === P0: LARGE VISUAL COLOR SWATCHES FOR REGIONS ===
        st.markdown("---")
        st.markdown("### 📍 Analyzed Regions")
        
        col1, col2, col3 = st.columns(3)
        
        regions_data = [
            ('left_cheek', 'Left Cheek', col1),
            ('right_cheek', 'Right Cheek', col2),
            ('forehead', 'Forehead', col3)
        ]
        
        for region_key, region_label, col in regions_data:
            with col:
                hex_val = results[region_key]['after']['hex']
                st.markdown(f"**{region_label}**")
                st.markdown(f'''
                <div style="text-align: center;">
                    <div style="
                        width: 100%; 
                        height: 120px; 
                        background-color: {hex_val}; 
                        margin: 10px auto; 
                        border-radius: 10px;
                        border: 3px solid #ddd;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
                    "></div>
                    <div style="font-size: 16px; font-weight: bold; color: #444;">
                        {hex_val.upper()}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
        
        # Show clean original image
        st.markdown("---")
        st.markdown("### 📷 Your Image")
        st.image(image_rgb, caption="Uploaded Image", use_container_width=True)
        
        # === P0: TECHNICAL DETAILS IN EXPANDER (COLLAPSED BY DEFAULT) ===
        with st.expander("🔧 Technical Details", expanded=False):
            st.markdown("#### How It Works")
            st.markdown("""
            This tool analyzes your skin tone by:
            1. Detecting facial landmarks using MediaPipe
            2. Sampling color from three regions: left cheek, right cheek, and forehead
            3. Applying CLAHE (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
            4. Averaging the colors to determine your overall skin tone
            """)
            
            # CLAHE Before/After Comparison
            st.markdown("#### CLAHE Normalization Comparison")
            viz, _ = visualize_skin_analysis_before_after(image, landmarks)
            st.image(
                cv2.cvtColor(viz, cv2.COLOR_BGR2RGB),
                caption="Before vs After CLAHE Enhancement",
                use_container_width=True
            )
            
            # Detailed color values
            st.markdown("#### Detailed Color Values")
            
            tab1, tab2 = st.tabs(["Before (Original)", "After (Normalized)"])
            
            with tab1:
                for region in ['left_cheek', 'right_cheek', 'forehead', 'final']:
                    region_label = region.replace('_', ' ').title()
                    hex_val = results[region]['before']['hex']
                    rgb_val = results[region]['before']['RGB']
                    st.markdown(f"**{region_label}:** `{hex_val}` RGB{rgb_val}")
            
            with tab2:
                for region in ['left_cheek', 'right_cheek', 'forehead', 'final']:
                    region_label = region.replace('_', ' ').title()
                    hex_val = results[region]['after']['hex']
                    rgb_val = results[region]['after']['RGB']
                    st.markdown(f"**{region_label}:** `{hex_val}` RGB{rgb_val}")
            
            # Debug overlays with landmarks
            st.markdown("#### Debug Visualization")
            dbg = draw_debug_overlays(
                image,
                result,
                landmarks,
                box_color=(0, 0, 255),   # red bbox
                lm_color=(255, 0, 0),    # blue landmarks
                poly_color=(0, 255, 255) # yellow polygons
            )
            st.image(cv2.cvtColor(dbg, cv2.COLOR_BGR2RGB),
                     caption="Debug: Face bounding box (red), landmarks (blue), sampled regions (yellow)")
        
        st.markdown("---")
        st.caption("💡 **Tip:** Expand 'Technical Details' above to see how the analysis works under the hood.")