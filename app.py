"""
Traffic Sign Recognition System
Multi-Module Streamlit Application powered by OpenCV, TensorFlow / Keras CNN, and YOLO.

Modules:
1. Home / Dashboard
2. Image Upload & Detection
3. Video & Live Camera Detection
4. Prediction Results & Sign Information
5. Detection History & Analytics
"""

import os
import sys
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Ensure base directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import (
    TRAFFIC_SIGN_CLASSES,
    CATEGORIES_LIST,
    CATEGORY_COLORS,
    get_class_info,
    CATEGORY_SPEED_LIMIT,
    CATEGORY_STOP_YIELD,
    CATEGORY_PROHIBITORY,
    CATEGORY_MANDATORY,
    CATEGORY_WARNING
)
from src.database import (
    log_detection,
    get_recent_detections,
    get_all_detections,
    get_dashboard_metrics,
    clear_detection_history
)
from src.detector import TrafficSignDetector
from src.utils import (
    image_to_base64,
    get_dark_navy_css,
    create_category_pie_chart,
    create_confidence_bar_chart,
    create_confidence_distribution_chart
)
from src.sample_data_generator import draw_traffic_sign

# Set Streamlit Page Config
st.set_page_config(
    page_title="Traffic Sign Recognition System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Dark Navy-Blue Road-Themed Styles
st.markdown(get_dark_navy_css(), unsafe_allow_html=True)

# Initialize Session State
if "current_module" not in st.session_state:
    st.session_state["current_module"] = "Home / Dashboard"
if "last_detection_result" not in st.session_state:
    st.session_state["last_detection_result"] = None
if "selected_sign_info" not in st.session_state:
    st.session_state["selected_sign_info"] = None

@st.cache_resource
def get_detector():
    """Cache the TrafficSignDetector engine."""
    return TrafficSignDetector(confidence_threshold=0.55)

detector = get_detector()

# ==========================================
# SIDEBAR NAVIGATION & TELEMETRY
# ==========================================
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 15px 0 10px 0;">
            <div style="font-size: 2.8rem; margin-bottom: 5px;">🚦</div>
            <h2 style="font-size: 1.3rem; font-weight: 800; margin: 0; color: #FFFFFF; letter-spacing: -0.02em;">TSR ADAS VISION</h2>
            <p style="font-size: 0.75rem; color: #38BDF8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;">Traffic Sign AI System</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 10px 0 20px 0;'>", unsafe_allow_html=True)
    
    # Navigation Modules
    modules = [
        "Home / Dashboard",
        "Image Upload & Detection",
        "Video & Live Camera Detection",
        "Prediction Results & Sign Info",
        "Detection History & Analytics"
    ]
    
    # Check if a navigation button was clicked in main page
    if "nav_target" in st.session_state and st.session_state["nav_target"] in modules:
        st.session_state["current_module"] = st.session_state["nav_target"]
        del st.session_state["nav_target"]
        
    choice = st.radio(
        "Navigation",
        modules,
        index=modules.index(st.session_state["current_module"]),
        label_visibility="collapsed"
    )
    st.session_state["current_module"] = choice

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 25px 0 15px 0;'>", unsafe_allow_html=True)
    
    # Engine Telemetry Widget
    st.markdown("""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 14px;">
            <div style="font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94A3B8; font-weight: 700; margin-bottom: 8px;">ENGINE TELEMETRY</div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 5px;">
                <span style="color: #94A3B8;">Detection:</span>
                <span style="color: #38BDF8; font-weight: 600;">YOLO / OpenCV RoI</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 5px;">
                <span style="color: #94A3B8;">Classifier:</span>
                <span style="color: #10B981; font-weight: 600;">Deep CNN (43-Class)</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 5px;">
                <span style="color: #94A3B8;">Backend:</span>
                <span style="color: #F8FAFC; font-weight: 600;">TensorFlow / Keras</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <span style="color: #94A3B8;">Inference:</span>
                <span style="color: #FCD34D; font-weight: 600;">Real-Time CPU/GPU</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; font-size: 0.7rem; color: #64748B; margin-top: 25px;">
            Traffic Sign Recognition System v2.0<br>Autonomous Driving Assistance
        </div>
    """, unsafe_allow_html=True)


# ==========================================
# MODULE 1: HOME / DASHBOARD
# ==========================================
if st.session_state["current_module"] == "Home / Dashboard":
    # Hero Banner
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-title">Traffic Sign Recognition System</div>
            <div class="hero-subtitle">
                Automated road survey and intelligent driver assistance platform. Combines OpenCV computer vision,
                YOLO spatial candidate localization, and Deep Learning Convolutional Neural Networks (CNN) for
                high-precision traffic sign detection and driving compliance advisory.
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Key Statistics Cards
    metrics = get_dashboard_metrics()
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🖼️</div>
                <div class="metric-label">Total Images Analysed</div>
                <div class="metric-val">{metrics['total_images']}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-label">Total Signs Detected</div>
                <div class="metric-val">{metrics['total_signs']}</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">✨</div>
                <div class="metric-label">Average Confidence</div>
                <div class="metric-val">{metrics['avg_confidence']}%</div>
            </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🎥</div>
                <div class="metric-label">Videos / Feeds Processed</div>
                <div class="metric-val">{metrics['total_videos']}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)

    # Quick Access Buttons
    st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;'>⚡ Quick Access Actions</h3>", unsafe_allow_html=True)
    qa1, qa2, qa3, qa4, qa5 = st.columns(5)
    
    with qa1:
        if st.button("📷 Upload Image", use_container_width=True):
            st.session_state["nav_target"] = "Image Upload & Detection"
            st.rerun()
    with qa2:
        if st.button("🎬 Video Detection", use_container_width=True):
            st.session_state["nav_target"] = "Video & Live Camera Detection"
            st.session_state["video_mode"] = "Video File"
            st.rerun()
    with qa3:
        if st.button("🔴 Live Camera", use_container_width=True):
            st.session_state["nav_target"] = "Video & Live Camera Detection"
            st.session_state["video_mode"] = "Live Camera"
            st.rerun()
    with qa4:
        if st.button("📘 Sign Guide", use_container_width=True):
            st.session_state["nav_target"] = "Prediction Results & Sign Info"
            st.rerun()
    with qa5:
        if st.button("📊 Analytics & History", use_container_width=True):
            st.session_state["nav_target"] = "Detection History & Analytics"
            st.rerun()

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # Display Traffic Sign Categories Section
    st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 14px;'>🚸 Monitored Traffic Sign Categories</h3>", unsafe_allow_html=True)
    
    cat_cols = st.columns(5)
    cat_definitions = [
        (CATEGORY_SPEED_LIMIT, "⚡", "Speed Limit", "Regulatory limits from 20 km/h to 120 km/h."),
        (CATEGORY_STOP_YIELD, "🛑", "Stop & Yield", "Complete stop octagons and priority yield signs."),
        (CATEGORY_PROHIBITORY, "⛔", "No Entry", "One-way restrictions, no entry, and passing bans."),
        (CATEGORY_MANDATORY, "🔄", "Turn & Mandatory", "Required turn directions, straight lanes, roundabouts."),
        (CATEGORY_WARNING, "⚠️", "Danger & Warning", "Pedestrians, road works, sharp curves, hazard alerts.")
    ]
    
    for i, (cat_key, icon, label, desc) in enumerate(cat_definitions):
        count = metrics["category_counts"].get(cat_key, 0)
        with cat_cols[i]:
            st.markdown(f"""
                <div class="cat-card">
                    <div class="cat-icon">{icon}</div>
                    <div class="cat-title">{label}</div>
                    <div class="cat-desc">{desc}</div>
                    <div style="margin-top: 10px; font-size: 0.8rem; font-weight: 700; color: #38BDF8;">
                        {count} Detections
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # Recently Detected Traffic Signs Carousel
    st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 14px;'>🕒 Recently Detected Traffic Signs</h3>", unsafe_allow_html=True)
    recent = get_recent_detections(limit=6)
    
    if recent:
        r_cols = st.columns(len(recent))
        for i, row in enumerate(recent):
            cid = row["class_id"]
            sign_img = draw_traffic_sign(cid, size=90)
            b64_img = image_to_base64(sign_img)
            cat = row["category"]
            badge_class = CATEGORY_COLORS.get(cat, {}).get("badge", "badge-warn")
            
            with r_cols[i]:
                st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 14px; text-align: center;">
                        <img src="data:image/png;base64,{b64_img}" style="width: 75px; height: 75px; border-radius: 8px; margin-bottom: 8px; object-fit: contain;">
                        <div style="font-weight: 700; font-size: 0.85rem; color: #F8FAFC; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;" title="{row['sign_name']}">{row['sign_name']}</div>
                        <div style="margin: 6px 0;">
                            <span class="sign-badge {badge_class}">{row['category']}</span>
                        </div>
                        <div style="font-size: 0.78rem; color: #10B981; font-weight: 700;">{row['confidence']}% Conf</div>
                        <div style="font-size: 0.7rem; color: #64748B; margin-top: 3px;">{row['timestamp'].split(' ')[1] if ' ' in row['timestamp'] else row['timestamp']}</div>
                    </div>
                """, unsafe_allow_html=True)


# ==========================================
# MODULE 2: IMAGE UPLOAD & DETECTION
# ==========================================
elif st.session_state["current_module"] == "Image Upload & Detection":
    st.markdown("<h2 style='font-weight: 800; color: #F8FAFC;'>📷 Road Image Detection</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; margin-top: -8px;'>Upload high-resolution road photographs or select curated test scenes. OpenCV extracts candidate regions, YOLO localizes bounds, and the Deep CNN classifies each traffic sign.</p>", unsafe_allow_html=True)

    col_input, col_preset = st.columns([1.2, 1])
    
    with col_input:
        uploaded_file = st.file_uploader(
            "Upload Road Traffic Image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
            help="Drag and drop or browse road photography"
        )
        
    with col_preset:
        st.markdown("<label style='font-size: 0.85rem; font-weight: 600; color: #E2E8F0;'>Or Choose a Preloaded Highway / Urban Scene:</label>", unsafe_allow_html=True)
        sample_options = {
            "Select a preloaded road image...": None,
            "Highway Arterial (Speed Limit 60 km/h)": "sample_highway_speed60.jpg",
            "Urban Intersection (Stop Sign)": "sample_urban_stop_sign.jpg",
            "Downtown One-Way (No Entry Restriction)": "sample_one_way_no_entry.jpg",
            "Roundabout Approach (Roundabout Mandatory & Yield)": "sample_roundabout_ahead.jpg",
            "Mountain Pass (Caution Warning & Turn Right)": "sample_mountain_warning.jpg",
            "School Zone Route (Speed Limit 30 km/h & Work Ahead)": "sample_school_zone_speed30.jpg"
        }
        selected_sample_label = st.selectbox("Sample Scenes", list(sample_options.keys()), label_visibility="collapsed")
        selected_sample_file = sample_options[selected_sample_label]

    active_image = None
    source_filename = "Uploaded Road Image"

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        active_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        source_filename = uploaded_file.name
    elif selected_sample_file:
        sample_path = os.path.join(BASE_DIR, "data", "samples", selected_sample_file)
        if os.path.exists(sample_path):
            active_image = cv2.imread(sample_path)
            source_filename = selected_sample_file

    if active_image is not None:
        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)
        
        c_detect_btn, c_conf_slider, _ = st.columns([1.2, 1.5, 2])
        with c_conf_slider:
            conf_threshold = st.slider("Detection Confidence Threshold", min_value=0.40, max_value=0.95, value=0.55, step=0.05)
        with c_detect_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            detect_clicked = st.button("🚀 Detect Traffic Signs", use_container_width=True)

        if detect_clicked or ("auto_detect" in st.session_state and st.session_state["auto_detect"]):
            with st.spinner("Processing image: OpenCV color RoI -> Deep CNN Classifier -> ADAS HUD Rendering..."):
                t_start = time.time()
                detections = detector.detect_and_classify(active_image, confidence_threshold=conf_threshold)
                t_elapsed = (time.time() - t_start) * 1000.0
                
                # Annotate image
                annotated_bgr = detector.draw_detections(active_image, detections)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                original_rgb = cv2.cvtColor(active_image, cv2.COLOR_BGR2RGB)
                
                # Save to session state for Module 4
                st.session_state["last_detection_result"] = {
                    "original_rgb": original_rgb,
                    "annotated_rgb": annotated_rgb,
                    "detections": detections,
                    "source_name": source_filename,
                    "elapsed_ms": t_elapsed
                }
                
                # Log each detection into SQLite DB
                for d in detections:
                    log_detection(
                        source_type="Image",
                        class_id=d["class_id"],
                        sign_name=d["name"],
                        category=d["category"],
                        confidence=d["confidence"],
                        source_name=source_filename,
                        bbox=d["bbox"]
                    )

        # Render Results if available
        if st.session_state["last_detection_result"] is not None:
            res = st.session_state["last_detection_result"]
            dets = res["detections"]
            
            st.markdown(f"""
                <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 12px; padding: 12px 18px; margin: 15px 0 20px 0; display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 700; color: #10B981; font-size: 1rem;">✅ Detection Complete:</span>
                        <span style="color: #F8FAFC; margin-left: 6px; font-weight: 600;">Found {len(dets)} traffic sign(s)</span>
                    </div>
                    <div class="hud-mono" style="color: #38BDF8;">
                        Latency: {res['elapsed_ms']:.1f}ms | Resolution: {res['original_rgb'].shape[1]}x{res['original_rgb'].shape[0]}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Side-by-side comparison
            tab1, tab2 = st.tabs(["🔍 Side-by-Side Comparison", "🖼️ Annotated Fullscreen View"])
            with tab1:
                col_orig, col_ann = st.columns(2)
                with col_orig:
                    st.markdown("<h4 style='font-size: 0.95rem; color: #94A3B8; text-align: center;'>Original Input Scene</h4>", unsafe_allow_html=True)
                    st.image(res["original_rgb"], use_container_width=True)
                with col_ann:
                    st.markdown("<h4 style='font-size: 0.95rem; color: #38BDF8; text-align: center;'>Detected Signs with HUD Bounding Boxes</h4>", unsafe_allow_html=True)
                    st.image(res["annotated_rgb"], use_container_width=True)
                    
            with tab2:
                st.image(res["annotated_rgb"], use_container_width=True)

            # Detected Signs Cards
            if dets:
                st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin: 25px 0 12px 0;'>🎯 Detected Sign Details</h3>", unsafe_allow_html=True)
                card_cols = st.columns(min(len(dets), 4))
                for idx, d in enumerate(dets):
                    with card_cols[idx % 4]:
                        crop_b64 = image_to_base64(d["crop_rgb"])
                        cat = d["category"]
                        badge_class = CATEGORY_COLORS.get(cat, {}).get("badge", "badge-warn")
                        
                        st.markdown(f"""
                            <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255,255,255,0.1); border-radius: 14px; padding: 16px; text-align: center;">
                                <img src="data:image/png;base64,{crop_b64}" style="width: 85px; height: 85px; border-radius: 8px; margin-bottom: 10px; object-fit: contain; background: #1E293B; border: 1px solid rgba(255,255,255,0.05);">
                                <div style="font-weight: 700; color: #FFFFFF; font-size: 0.95rem;">{d['name']}</div>
                                <div style="margin: 6px 0;"><span class="sign-badge {badge_class}">{d['category']}</span></div>
                                <div style="font-size: 0.85rem; color: #10B981; font-weight: 700;">Confidence: {d['confidence']}%</div>
                                <div class="hud-mono" style="font-size: 0.72rem; color: #94A3B8; margin-top: 4px;">BBox: {d['bbox']}</div>
                            </div>
                        """, unsafe_allow_html=True)
                        if st.button(f"📘 View Driving Advice", key=f"view_info_{idx}", use_container_width=True):
                            st.session_state["selected_sign_info"] = d
                            st.session_state["nav_target"] = "Prediction Results & Sign Info"
                            st.rerun()
            else:
                st.info("No traffic signs detected matching the specified confidence threshold. Try lowering the threshold or selecting another image.")
    else:
        # Placeholder instructions
        st.markdown("""
            <div style="border: 2px dashed rgba(255,255,255,0.15); border-radius: 16px; padding: 45px; text-align: center; margin-top: 20px;">
                <div style="font-size: 3rem; margin-bottom: 10px;">🛣️</div>
                <h3 style="color: #F8FAFC; font-weight: 700;">No Road Scene Selected</h3>
                <p style="color: #94A3B8; max-width: 500px; margin: 0 auto;">
                    Upload a photograph from your camera roll or pick one of the preloaded highway and city scenes above to test traffic sign detection.
                </p>
            </div>
        """, unsafe_allow_html=True)


# ==========================================
# MODULE 3: VIDEO & LIVE CAMERA DETECTION
# ==========================================
elif st.session_state["current_module"] == "Video & Live Camera Detection":
    st.markdown("<h2 style='font-weight: 800; color: #F8FAFC;'>🎬 Video & Live Stream Detection</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; margin-top: -8px;'>Analyze dynamic road footage frame-by-frame. OpenCV decodes video frames, extracts bounding boxes, tracks detected signs, and provides real-time telemetry.</p>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Source Selection",
        ["Preloaded Sample Dashcam Drive", "Upload Video File", "Live Camera Feed"],
        horizontal=True,
        label_visibility="collapsed"
    )

    video_source_path = None
    is_live = False

    if input_mode == "Preloaded Sample Dashcam Drive":
        sample_vid = os.path.join(BASE_DIR, "data", "samples", "sample_traffic_drive.mp4")
        if os.path.exists(sample_vid):
            video_source_path = sample_vid
            st.success("Loaded preloaded driving video clip: Approach passing Speed Limit 60, Stop Sign, and Turn Right.")
        else:
            st.error("Sample video clip not found. Generate it from the generator module.")

    elif input_mode == "Upload Video File":
        vid_upload = st.file_uploader("Upload MP4, AVI, or MOV Video", type=["mp4", "avi", "mov"])
        if vid_upload is not None:
            temp_vid_path = os.path.join(BASE_DIR, "data", "uploads", vid_upload.name)
            os.makedirs(os.path.dirname(temp_vid_path), exist_ok=True)
            with open(temp_vid_path, "wb") as f:
                f.write(vid_upload.read())
            video_source_path = temp_vid_path
            st.success(f"Video uploaded: {vid_upload.name}")

    elif input_mode == "Live Camera Feed":
        is_live = True
        st.info("Live camera mode activated. Stream camera frames to detect traffic signs in real time.")
        cam_snap = st.camera_input("Take Live Snapshot / Stream Frame")
        if cam_snap is not None:
            pil_snap = Image.open(cam_snap)
            snap_bgr = cv2.cvtColor(np.array(pil_snap), cv2.COLOR_RGB2BGR)
            
            with st.spinner("Analyzing snapshot..."):
                dets = detector.detect_and_classify(snap_bgr, confidence_threshold=0.50)
                ann_bgr = detector.draw_detections(snap_bgr, dets)
                ann_rgb = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2RGB)
                
                for d in dets:
                    log_detection(
                        source_type="Live Camera",
                        class_id=d["class_id"],
                        sign_name=d["name"],
                        category=d["category"],
                        confidence=d["confidence"],
                        source_name="Live Camera Stream",
                        bbox=d["bbox"]
                    )
                    
                st.image(ann_rgb, caption=f"Live Camera Detections: {len(dets)} sign(s)", use_container_width=True)
                if dets:
                    det_strings = [f"{d['name']} ({d['confidence']}%)" for d in dets]
                    st.write(f"Detected: {', '.join(det_strings)}")

    if video_source_path and not is_live:
        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)
        
        # Telemetry & Controls
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns([1, 1, 1.5])
        with c_ctrl1:
            frame_skip = st.selectbox("Process Every Nth Frame", [1, 2, 3], index=0, help="Higher skip speeds up playback on slower hardware")
        with c_ctrl2:
            conf_thresh_vid = st.slider("Min Confidence", 0.40, 0.90, 0.55, 0.05)
        with c_ctrl3:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            start_detection = st.button("▶️ Start Video Detection", use_container_width=True)

        if start_detection:
            cap = cv2.VideoCapture(video_source_path)
            total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 120
            video_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
            
            # UI display containers
            col_video_feed, col_telemetry = st.columns([2, 1.2])
            
            with col_video_feed:
                feed_placeholder = st.empty()
                progress_bar = st.progress(0)
                
            with col_telemetry:
                fps_metric = st.empty()
                signs_metric = st.empty()
                unique_metric = st.empty()
                st.markdown("<h4 style='font-size: 0.9rem; color: #94A3B8; margin-top: 15px;'>Live Rolling Detection Log</h4>", unsafe_allow_html=True)
                log_placeholder = st.empty()

            frame_idx = 0
            unique_detected = set()
            rolling_logs = []
            
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                if frame_idx % frame_skip != 0:
                    continue

                t0 = time.time()
                dets = detector.detect_and_classify(frame_bgr, confidence_threshold=conf_thresh_vid)
                t1 = time.time()
                proc_fps = 1.0 / max(0.001, (t1 - t0))

                ann_frame_bgr = detector.draw_detections(frame_bgr, dets)
                ann_frame_rgb = cv2.cvtColor(ann_frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Update feed
                feed_placeholder.image(ann_frame_rgb, use_container_width=True)
                progress_bar.progress(min(1.0, frame_idx / total_video_frames))

                # Track unique and log
                for d in dets:
                    unique_detected.add(d["name"])
                    rolling_logs.insert(0, f"Frame {frame_idx:03d} | {d['name']} ({d['confidence']}%)")
                    # Log periodically into DB
                    if frame_idx % 6 == 0:
                        log_detection(
                            source_type="Video",
                            class_id=d["class_id"],
                            sign_name=d["name"],
                            category=d["category"],
                            confidence=d["confidence"],
                            source_name=os.path.basename(video_source_path),
                            bbox=d["bbox"]
                        )
                        
                rolling_logs = rolling_logs[:8]

                # Update telemetry
                fps_metric.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
                        <span style="color: #94A3B8; font-size: 0.8rem;">Processing Speed:</span>
                        <span style="color: #38BDF8; font-weight: 700; float: right;">{proc_fps:.1f} FPS</span>
                    </div>
                """, unsafe_allow_html=True)
                
                signs_metric.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
                        <span style="color: #94A3B8; font-size: 0.8rem;">Current Signs in Frame:</span>
                        <span style="color: #10B981; font-weight: 700; float: right;">{len(dets)}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                unique_metric.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px;">
                        <span style="color: #94A3B8; font-size: 0.8rem;">Unique Signs Identified:</span>
                        <span style="color: #FCD34D; font-weight: 700; float: right;">{len(unique_detected)}</span>
                    </div>
                """, unsafe_allow_html=True)

                log_items_html = "".join([f"<li style='color: #E2E8F0; font-size: 0.8rem; margin-bottom: 4px;' class='hud-mono'>{item}</li>" for item in rolling_logs])
                log_placeholder.markdown(f"<ul style='list-style: none; padding-left: 0;'>{log_items_html}</ul>", unsafe_allow_html=True)

            cap.release()
            st.success(f"Video analysis complete! Identified {len(unique_detected)} unique traffic signs across {frame_idx} processed frames.")


# ==========================================
# MODULE 4: PREDICTION RESULTS & SIGN INFO
# ==========================================
elif st.session_state["current_module"] == "Prediction Results & Sign Info":
    st.markdown("<h2 style='font-weight: 800; color: #F8FAFC;'>📘 Prediction Results & Sign Knowledge</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; margin-top: -8px;'>Detailed diagnostic view of detected traffic signs with driving compliance rules, legal penalty risks, and the full interactive sign encyclopedia.</p>", unsafe_allow_html=True)

    # If a sign was selected from Module 2
    active_det = st.session_state.get("selected_sign_info")
    if active_det is None and st.session_state.get("last_detection_result"):
        dets = st.session_state["last_detection_result"]["detections"]
        if dets:
            active_det = dets[0]
            
    if active_det is not None:
        st.markdown("<h3 style='font-size: 1.2rem; font-weight: 700; color: #38BDF8; margin-bottom: 12px;'>🔍 Deep Sign Inspection</h3>", unsafe_allow_html=True)
        
        info_c1, info_c2 = st.columns([1, 2])
        
        with info_c1:
            crop_b64 = image_to_base64(active_det["crop_rgb"])
            st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 16px; padding: 25px; text-align: center;">
                    <img src="data:image/png;base64,{crop_b64}" style="width: 140px; height: 140px; border-radius: 12px; margin-bottom: 15px; object-fit: contain; background: #0B1120; border: 1px solid rgba(255,255,255,0.1);">
                    <div style="font-size: 1.2rem; font-weight: 800; color: #FFFFFF;">{active_det['name']}</div>
                    <div style="font-size: 0.9rem; color: #38BDF8; font-weight: 600; margin-top: 4px;">{active_det['category']}</div>
                    <div style="margin-top: 15px; background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 8px;">
                        <span style="color: #94A3B8; font-size: 0.8rem;">Confidence:</span>
                        <span style="color: #10B981; font-weight: 800; font-size: 1.1rem; margin-left: 6px;">{active_det['confidence']}%</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with info_c2:
            st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 22px;">
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.78rem; text-transform: uppercase; color: #94A3B8; font-weight: 700; letter-spacing: 0.05em;">SIGN MEANING</div>
                        <div style="font-size: 1.05rem; color: #F8FAFC; font-weight: 600; margin-top: 3px;">{active_det['meaning']}</div>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.78rem; text-transform: uppercase; color: #38BDF8; font-weight: 700; letter-spacing: 0.05em;">SAFE DRIVING INSTRUCTION</div>
                        <div style="font-size: 0.95rem; color: #E2E8F0; margin-top: 3px; line-height: 1.5;">{active_det['instruction']}</div>
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px;">
                        <div style="background: rgba(30, 41, 59, 0.5); padding: 12px; border-radius: 10px;">
                            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">PRIORITY LEVEL</div>
                            <div style="font-size: 0.9rem; color: #FCD34D; font-weight: 600; margin-top: 2px;">{active_det['priority']}</div>
                        </div>
                        <div style="background: rgba(30, 41, 59, 0.5); padding: 12px; border-radius: 10px;">
                            <div style="font-size: 0.75rem; color: #94A3B8; font-weight: 700;">LEGAL COMPLIANCE RISK</div>
                            <div style="font-size: 0.9rem; color: #F87171; font-weight: 600; margin-top: 2px;">{active_det['penalty_risk']}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 30px 0;'>", unsafe_allow_html=True)

    # Interactive Traffic Sign Encyclopedia
    st.markdown("<h3 style='font-size: 1.25rem; font-weight: 700; color: #F8FAFC;'>📚 Comprehensive Traffic Sign Guide (GTSRB 43 Classes)</h3>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8;'>Filter and explore standard international road signs, visual appearances, legal meanings, and safe driver behaviors.</p>", unsafe_allow_html=True)

    guide_filter_cat = st.selectbox(
        "Filter Guide by Category",
        ["All Categories"] + CATEGORIES_LIST
    )
    
    # Filter classes
    displayed_classes = []
    for cid, meta in TRAFFIC_SIGN_CLASSES.items():
        if guide_filter_cat == "All Categories" or meta["category"] == guide_filter_cat:
            displayed_classes.append((cid, meta))
            
    cols_per_row = 3
    for row_idx in range(0, len(displayed_classes), cols_per_row):
        row_items = displayed_classes[row_idx:row_idx + cols_per_row]
        cols = st.columns(cols_per_row)
        for c_idx, (cid, meta) in enumerate(row_items):
            with cols[c_idx]:
                sign_img = draw_traffic_sign(cid, size=80)
                b64 = image_to_base64(sign_img)
                badge_class = CATEGORY_COLORS.get(meta["category"], {}).get("badge", "badge-warn")
                
                st.markdown(f"""
                    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px; margin-bottom: 16px;">
                        <div style="display: flex; align-items: center; margin-bottom: 10px;">
                            <img src="data:image/png;base64,{b64}" style="width: 55px; height: 55px; border-radius: 6px; margin-right: 12px; object-fit: contain;">
                            <div>
                                <div style="font-weight: 700; font-size: 0.92rem; color: #FFFFFF;">{meta['name']}</div>
                                <span class="sign-badge {badge_class}" style="margin-top: 4px;">{meta['category']}</span>
                            </div>
                        </div>
                        <div style="font-size: 0.8rem; color: #CBD5E1; margin-bottom: 6px;">
                            <strong>Meaning:</strong> {meta['meaning']}
                        </div>
                        <div style="font-size: 0.78rem; color: #94A3B8;">
                            <strong>Instruction:</strong> {meta['instruction']}
                        </div>
                    </div>
                """, unsafe_allow_html=True)


# ==========================================
# MODULE 5: DETECTION HISTORY & ANALYTICS
# ==========================================
elif st.session_state["current_module"] == "Detection History & Analytics":
    st.markdown("<h2 style='font-weight: 800; color: #F8FAFC;'>📊 Detection History & Analytics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94A3B8; margin-top: -8px;'>Audit past road surveys, filter records by confidence and category, examine analytical distribution charts, and export data.</p>", unsafe_allow_html=True)

    # Filter Controls Bar
    with st.container():
        f1, f2, f3, f4 = st.columns([1.5, 1, 1, 1.2])
        with f1:
            search_query = st.text_input("🔍 Search Sign Name or Source", placeholder="e.g. Speed Limit, Stop, highway...")
        with f2:
            cat_choice = st.selectbox("Category Filter", ["All"] + CATEGORIES_LIST)
        with f3:
            src_choice = st.selectbox("Source Filter", ["All", "Image", "Video", "Live Camera"])
        with f4:
            min_conf = st.slider("Min Confidence (%)", 0.0, 95.0, 0.0, 5.0)

    # Retrieve matching records from SQLite
    df_records = get_all_detections(
        search_query=search_query,
        category_filter=cat_choice,
        source_filter=src_choice,
        min_confidence=min_conf,
        limit=500
    )

    # Top KPI summary for filtered data
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Records Found", len(df_records))
    with kpi2:
        avg_c = round(df_records["confidence"].mean(), 1) if not df_records.empty else 0.0
        st.metric("Average Confidence", f"{avg_c}%")
    with kpi3:
        unique_cnt = df_records["sign_name"].nunique() if not df_records.empty else 0
        st.metric("Unique Signs", unique_cnt)
    with kpi4:
        sources_cnt = df_records["source_type"].nunique() if not df_records.empty else 0
        st.metric("Input Modes", sources_cnt)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Visual Analytics Charts
    if not df_records.empty:
        st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;'>📈 Visual Trend Analytics</h3>", unsafe_allow_html=True)
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("<h5 style='font-size: 0.88rem; color: #94A3B8;'>Detections by Sign Category</h5>", unsafe_allow_html=True)
            cat_counts = df_records["category"].value_counts().to_dict()
            fig_pie = create_category_pie_chart(cat_counts)
            if fig_pie:
                st.plotly_chart(fig_pie, use_container_width=True)
                
        with chart_col2:
            st.markdown("<h5 style='font-size: 0.88rem; color: #94A3B8;'>Top Detected Traffic Signs</h5>", unsafe_allow_html=True)
            fig_bar = create_confidence_bar_chart(df_records)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<h5 style='font-size: 0.88rem; color: #94A3B8;'>Confidence Score Distribution</h5>", unsafe_allow_html=True)
        fig_dist = create_confidence_distribution_chart(df_records)
        if fig_dist:
            st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown("<hr style='border: 0; border-top: 1px solid rgba(255,255,255,0.08); margin: 25px 0 15px 0;'>", unsafe_allow_html=True)

    # Detailed Records Table
    st.markdown("<h3 style='font-size: 1.15rem; font-weight: 700; color: #F8FAFC; margin-bottom: 12px;'>📋 Audit History Log</h3>", unsafe_allow_html=True)
    
    if not df_records.empty:
        display_df = df_records[["id", "timestamp", "source_type", "source_name", "sign_name", "category", "confidence", "bbox_x1", "bbox_y1", "bbox_w", "bbox_h"]].copy()
        display_df.rename(columns={
            "id": "ID",
            "timestamp": "Timestamp",
            "source_type": "Source",
            "source_name": "File / Source",
            "sign_name": "Sign Detected",
            "category": "Category",
            "confidence": "Confidence (%)",
            "bbox_x1": "X",
            "bbox_y1": "Y",
            "bbox_w": "Width",
            "bbox_h": "Height"
        }, inplace=True)
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No detection records found matching your active filter criteria.")

    # Export & Clear Actions
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    act1, act2, act3 = st.columns([1, 1, 1.5])
    
    with act1:
        if not df_records.empty:
            csv_data = df_records.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export to CSV",
                data=csv_data,
                file_name=f"traffic_sign_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with act2:
        if not df_records.empty:
            json_data = df_records.to_json(orient="records").encode("utf-8")
            st.download_button(
                "📥 Export to JSON",
                data=json_data,
                file_name=f"traffic_sign_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
    with act3:
        if st.button("🗑️ Clear History Records", use_container_width=True):
            clear_detection_history()
            st.success("All detection records have been cleared from the database.")
            st.rerun()
