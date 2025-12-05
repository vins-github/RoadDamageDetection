"""
🚗 Road Damage Detection - Production Ready
Upload Video/Image + Real-time Processing (Cloud Compatible)
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .alert-box {
        background: #ff4444;
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        animation: blink 1s infinite;
        box-shadow: 0 0 30px rgba(255, 68, 68, 0.6);
        margin: 1rem 0;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
        border-radius: 10px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Class info
CLASS_INFO = {
    'Longitudinal_Crack': ('🔹 Retak Memanjang', '#FF6B6B'),
    'Transverse_Crack': ('↔️ Retak Melintang', '#4ECDC4'),
    'Alligator_Crack': ('🐊 Retak Buaya', '#95E1D3'),
    'Pothole': ('🕳️ Lubang', '#FFE66D')
}

# Header
st.markdown('<h1 class="main-header">🛣️ Road Damage Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📤 Upload & Detect | 🎯 YOLOv8 Real-time Processing</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Input mode
    st.markdown("### 📥 Input Mode")
    input_mode = st.radio(
        "Select Input Type",
        ["📹 Video File (MP4/AVI)", "🖼️ Image File (JPG/PNG)", "🎬 Demo Mode (Loop)"],
        help="Choose your input source"
    )
    
    # File uploader
    uploaded_file = None
    if input_mode == "📹 Video File (MP4/AVI)":
        uploaded_file = st.file_uploader(
            "Upload dashcam video",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload video rekaman dashcam"
        )
    elif input_mode == "🖼️ Image File (JPG/PNG)":
        uploaded_file = st.file_uploader(
            "Upload road image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload foto jalan rusak"
        )
    
    # Model settings
    st.markdown("### 🎯 Detection Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Semakin tinggi, semakin strict deteksi"
    )
    
    # Processing settings
    st.markdown("### ⚡ Processing")
    if input_mode == "📹 Video File (MP4/AVI)" or input_mode == "🎬 Demo Mode (Loop)":
        process_every_n_frames = st.slider(
            "Process every N frames",
            min_value=1,
            max_value=10,
            value=2,
            help="Skip frames untuk speed up (1 = every frame)"
        )
        loop_video = st.checkbox("Loop video", value=True, help="Ulangi video terus")
    
    # Display options
    st.markdown("### 🎨 Display Options")
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)
    show_fps = st.checkbox("Show FPS", value=True)
    
    st.markdown("---")
    st.markdown("""
    **💡 Tips:**
    - Upload video dashcam untuk demo
    - Format support: MP4, AVI, MOV
    - Bisa loop otomatis
    - Image untuk single detection
    """)
    
    st.markdown("---")
    st.success("✅ Running on Streamlit Cloud")

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False

# Main layout
col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown("### 📹 Detection Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    
    # Control buttons for video
    if input_mode in ["📹 Video File (MP4/AVI)", "🎬 Demo Mode (Loop)"]:
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("▶️ START", use_container_width=True, disabled=st.session_state.processing):
                st.session_state.processing = True
                st.rerun()
        with btn_col2:
            if st.button("⏹️ STOP", use_container_width=True, disabled=not st.session_state.processing):
                st.session_state.processing = False
                st.rerun()

with col2:
    st.markdown("### 📊 Live Statistics")
    fps_placeholder = st.empty()
    total_placeholder = st.empty()
    pothole_placeholder = st.empty()
    
    st.markdown("### 🏷️ Detections")
    detection_placeholder = st.empty()

# Process based on mode
if input_mode == "🖼️ Image File (JPG/PNG)" and uploaded_file:
    # Single image detection
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Run detection
    with st.spinner("🔍 Detecting..."):
        results = model.predict(img_array, conf=confidence, verbose=False)
        annotated = results[0].plot(labels=show_labels, conf=show_conf)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Count detections
    detections = {}
    pothole_detected = False
    
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        detections[cls_name] = detections.get(cls_name, 0) + 1
        if cls_name == 'Pothole':
            pothole_detected = True
    
    # Display results
    video_placeholder.image(annotated, channels="RGB", use_container_width=True)
    
    # Alert
    if pothole_detected:
        alert_placeholder.markdown(
            '<div class="alert-box">⚠️ POTHOLE DETECTED! ⚠️</div>',
            unsafe_allow_html=True
        )
    
    # Statistics
    with col2:
        total_placeholder.markdown(f"""
        <div class="metric-card">
            <h2 style="margin:0;">🎯 Total Detected</h2>
            <h1 style="margin:0.5rem 0;">{sum(detections.values())}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if 'Pothole' in detections:
            pothole_placeholder.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                <h2 style="margin:0;">🕳️ Potholes</h2>
                <h1 style="margin:0.5rem 0;">{detections['Pothole']}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Detection details
        if detections:
            details_html = "<div style='margin-top: 1rem;'>"
            for cls, count in detections.items():
                info = CLASS_INFO.get(cls, (cls, '#999'))
                details_html += f"""
                <div style="
                    background: {info[1]}22;
                    border-left: 4px solid {info[1]};
                    padding: 1rem;
                    margin: 0.5rem 0;
                    border-radius: 8px;
                ">
                    <strong style="font-size: 1.1rem;">{info[0]}</strong><br>
                    <span style="font-size: 1.5rem; font-weight: bold;">{count}x</span>
                </div>
                """
            details_html += "</div>"
            detection_placeholder.markdown(details_html, unsafe_allow_html=True)
        else:
            detection_placeholder.success("✅ No damage detected!")

elif (input_mode == "📹 Video File (MP4/AVI)" and uploaded_file) or input_mode == "🎬 Demo Mode (Loop)":
    # Video processing
    if st.session_state.processing:
        # Save uploaded video to temp file
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            # Demo mode - would need a sample video
            video_placeholder.warning("⚠️ Demo mode requires a sample video file")
            st.session_state.processing = False
            st.rerun()
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Cannot open video file")
            st.session_state.processing = False
        else:
            # Video info
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            
            frame_count = 0
            processed_count = 0
            start_time = time.time()
            total_detected = 0
            pothole_count = 0
            
            while st.session_state.processing:
                ret, frame = cap.read()
                
                # Loop video if enabled
                if not ret:
                    if loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        st.session_state.processing = False
                        break
                
                frame_count += 1
                
                # Process every N frames
                if frame_count % process_every_n_frames != 0:
                    continue
                
                processed_count += 1
                
                # Run detection
                results = model.predict(frame, conf=confidence, verbose=False, imgsz=640)
                annotated = results[0].plot(labels=show_labels, conf=show_conf)
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                # Count detections
                detections = {}
                pothole_detected = False
                
                for box in results[0].boxes:
                    cls_name = model.names[int(box.cls[0])]
                    detections[cls_name] = detections.get(cls_name, 0) + 1
                    if cls_name == 'Pothole':
                        pothole_detected = True
                
                total_detected += sum(detections.values())
                if pothole_detected:
                    pothole_count += 1
                
                # Calculate FPS
                elapsed = time.time() - start_time
                processing_fps = processed_count / elapsed if elapsed > 0 else 0
                
                # Add FPS to frame
                if show_fps:
                    cv2.rectangle(annotated, (10, 10), (250, 70), (0, 0, 0), -1)
                    cv2.putText(annotated, f"FPS: {processing_fps:.1f}", (20, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                # Display
                video_placeholder.image(annotated, channels="RGB", use_container_width=True)
                
                # Alert
                if pothole_detected:
                    alert_placeholder.markdown(
                        '<div class="alert-box">⚠️ POTHOLE AHEAD! ⚠️</div>',
                        unsafe_allow_html=True
                    )
                else:
                    alert_placeholder.empty()
                
                # Update statistics
                with col2:
                    fps_placeholder.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0;">⚡ Processing FPS</h3>
                        <h1 style="margin:0.5rem 0;">{processing_fps:.1f}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    total_placeholder.markdown(f"""
                    <div class="metric-card">
                        <h3 style="margin:0;">🎯 Total Detected</h3>
                        <h1 style="margin:0.5rem 0;">{total_detected}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    pothole_placeholder.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);">
                        <h3 style="margin:0;">🕳️ Pothole Frames</h3>
                        <h1 style="margin:0.5rem 0;">{pothole_count}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Current detections
                    if detections:
                        details_html = "<div style='margin-top: 1rem;'>"
                        for cls, count in detections.items():
                            info = CLASS_INFO.get(cls, (cls, '#999'))
                            details_html += f"""
                            <div style="
                                background: {info[1]}22;
                                border-left: 4px solid {info[1]};
                                padding: 0.8rem;
                                margin: 0.5rem 0;
                                border-radius: 8px;
                            ">
                                <strong>{info[0]}</strong><br>
                                <span style="font-size: 1.3rem;">{count}x</span>
                            </div>
                            """
                        details_html += "</div>"
                        detection_placeholder.markdown(details_html, unsafe_allow_html=True)
                    else:
                        detection_placeholder.info("ℹ️ No damage in current frame")
                
                # Small delay
                time.sleep(0.03)
            
            cap.release()
            if uploaded_file:
                os.unlink(video_path)
            
            alert_placeholder.success("✅ Processing completed!")
    else:
        video_placeholder.info("👆 Click START button to begin video processing")

else:
    video_placeholder.info(f"📤 Please upload a file to start detection")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p style="font-size: 1.1rem;"><b>🎓 Road Damage Detection System</b></p>
    <p>YOLOv8 Computer Vision | Real-time Processing | Cloud Deployment</p>
    <p style="font-size: 0.9rem; color: #999;">
        Developed for Deep Learning & Computer Vision Course
    </p>
</div>
""", unsafe_allow_html=True)
