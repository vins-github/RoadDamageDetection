"""
🚗 Road Damage Detection - Cloud Real-time Demo
Auto-loop demo video untuk simulasi real-time tanpa webcam!
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import tempfile
import urllib.request
import os

# Try to initialize pygame
try:
    import pygame
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
except (ImportError, Exception):
    AUDIO_AVAILABLE = False

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
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        animation: blink 1s infinite;
        box-shadow: 0 0 20px rgba(255, 68, 68, 0.5);
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .live-badge {
        background: #ff0000;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

# Generate beep
def generate_beep():
    if not AUDIO_AVAILABLE:
        return
    try:
        sample_rate = 22050
        duration = 0.2
        frequency = 440
        samples = int(sample_rate * duration)
        wave = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
        wave = (wave * 32767).astype(np.int16)
        stereo_wave = np.column_stack((wave, wave))
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
    except:
        pass

# Download demo video from URL (optional)
@st.cache_data
def download_demo_video(url):
    """Download demo video from URL"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        urllib.request.urlretrieve(url, temp_file.name)
        return temp_file.name
    except:
        return None

model = load_model()

# Class info
CLASS_INFO = {
    'Longitudinal_Crack': ('🔹 Retak Memanjang', '#FF6B6B', False),
    'Transverse_Crack': ('↔️ Retak Melintang', '#4ECDC4', False),
    'Alligator_Crack': ('🐊 Retak Buaya', '#95E1D3', False),
    'Pothole': ('🕳️ Lubang', '#FFE66D', True)
}

# Header
st.markdown('<h1 class="main-header">🛣️ Road Damage Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📹 Real-time Demo Mode | Upload Your Own Video</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Input mode
    st.markdown("### 📥 Demo Mode")
    demo_mode = st.radio(
        "Select Mode",
        [
            "🎬 Live Demo (Auto-loop)",
            "📤 Upload Your Video",
            "🖼️ Upload Image"
        ],
        help="Live Demo: Upload video sekali, akan loop seperti live stream"
    )
    
    uploaded_file = None
    demo_video_path = None
    
    if demo_mode == "🎬 Live Demo (Auto-loop)":
        st.info("📹 Upload video dashcam untuk demo real-time")
        uploaded_file = st.file_uploader(
            "Upload Demo Video",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Video ini akan di-loop terus seperti live stream"
        )
        
        # Option to use sample video URL
        use_sample = st.checkbox("Atau pakai sample video online", value=False)
        if use_sample:
            sample_url = st.text_input(
                "URL Video (YouTube, Google Drive, dll)",
                placeholder="https://example.com/dashcam.mp4",
                help="Paste URL video dashcam"
            )
            if sample_url and st.button("Download & Use"):
                with st.spinner("Downloading video..."):
                    demo_video_path = download_demo_video(sample_url)
                    if demo_video_path:
                        st.success("✅ Video downloaded!")
                    else:
                        st.error("❌ Failed to download")
    
    elif demo_mode == "📤 Upload Your Video":
        uploaded_file = st.file_uploader(
            "Choose Video File",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm']
        )
    
    else:  # Image mode
        uploaded_file = st.file_uploader(
            "Choose Image File",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
    
    # Model settings
    st.markdown("### 🎯 Model Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.05
    )
    
    # Audio settings
    st.markdown("### 🔊 Alert Settings")
    enable_audio = st.checkbox("Enable Audio Alert", value=AUDIO_AVAILABLE, disabled=not AUDIO_AVAILABLE)
    
    alert_cooldown = st.slider(
        "Alert Cooldown (seconds)",
        min_value=1,
        max_value=10,
        value=3
    )
    
    # Display options
    st.markdown("### 🎨 Display")
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)
    show_fps = st.checkbox("Show FPS", value=True)
    
    if demo_mode == "🎬 Live Demo (Auto-loop)":
        show_live_badge = st.checkbox("Show LIVE Badge", value=True)
    else:
        show_live_badge = False
    
    st.markdown("---")
    st.markdown("""
    **💡 Tips Demo:**
    - **Live Demo**: Video loop terus (seperti live)
    - Paling cocok untuk presentasi
    - Terlihat seperti real-time!
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    # Show LIVE badge if in demo mode
    if demo_mode == "🎬 Live Demo (Auto-loop)" and show_live_badge:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <span class="live-badge">🔴 LIVE</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📹 Detection Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Statistics")
    fps_metric = st.empty()
    detected_metric = st.empty()
    
    st.markdown("### 🏷️ Detections")
    detection_details = st.empty()

# Process single image
if demo_mode == "🖼️ Upload Image" and uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    with st.spinner("🔍 Analyzing..."):
        results = model.predict(img_array, conf=confidence, verbose=False)
        annotated = results[0].plot(labels=show_labels, conf=show_conf)
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        detections = {}
        pothole_detected = False
        
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            detections[cls_name] = detections.get(cls_name, 0) + 1
            if cls_name == 'Pothole':
                pothole_detected = True
        
        if pothole_detected:
            alert_placeholder.markdown(
                '<div class="alert-box">⚠️ POTHOLE DETECTED! ⚠️</div>',
                unsafe_allow_html=True
            )
            if enable_audio and AUDIO_AVAILABLE:
                generate_beep()
        
        video_placeholder.image(annotated, channels="RGB", use_container_width=True)
        
        with col2:
            detected_metric.metric("🎯 Total", sum(detections.values()))
            
            if detections:
                details_html = ""
                for cls, count in detections.items():
                    info = CLASS_INFO.get(cls, (cls, '#999', False))
                    details_html += f"""
                    <div style="
                        background: {info[1]}22;
                        border-left: 4px solid {info[1]};
                        padding: 0.5rem;
                        margin: 0.5rem 0;
                        border-radius: 5px;
                    ">
                        <strong>{info[0]}</strong><br>
                        {count}x
                    </div>
                    """
                detection_details.markdown(details_html, unsafe_allow_html=True)
            else:
                detection_details.success("✅ No damage")

# Process video
elif (uploaded_file or demo_video_path) and demo_mode != "🖼️ Upload Image":
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        start_btn = st.button("▶️ Start Detection", use_container_width=True, type="primary")
    with col_btn2:
        stop_btn = st.button("⏹️ Stop Detection", use_container_width=True)
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    
    if st.session_state.running:
        # Save uploaded file to temp
        if uploaded_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name
        else:
            video_path = demo_video_path
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("❌ Cannot open video!")
        else:
            if demo_mode == "🎬 Live Demo (Auto-loop)":
                st.success("🔴 LIVE: Auto-loop mode active!")
            else:
                st.success("✅ Video loaded!")
            
            frame_count = 0
            start_time = time.time()
            total_detected = 0
            last_alert_time = 0
            loop_count = 0
            
            while st.session_state.running:
                ret, frame = cap.read()
                
                # Auto-loop for demo mode
                if not ret:
                    if demo_mode == "🎬 Live Demo (Auto-loop)":
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        loop_count += 1
                        continue
                    else:
                        st.info("⏹️ Video ended")
                        break
                
                results = model.predict(frame, conf=confidence, verbose=False, imgsz=640)
                annotated = results[0].plot(labels=show_labels, conf=show_conf)
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                
                detections = {}
                pothole_detected = False
                
                for box in results[0].boxes:
                    cls_name = model.names[int(box.cls[0])]
                    detections[cls_name] = detections.get(cls_name, 0) + 1
                    if cls_name == 'Pothole':
                        pothole_detected = True
                
                total_detected += sum(detections.values())
                
                current_time = time.time()
                if pothole_detected and current_time - last_alert_time > alert_cooldown:
                    if enable_audio and AUDIO_AVAILABLE:
                        generate_beep()
                    last_alert_time = current_time
                
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                if show_fps:
                    cv2.rectangle(annotated, (10, 10), (200, 60), (0, 0, 0), -1)
                    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                # Show loop count in demo mode
                if demo_mode == "🎬 Live Demo (Auto-loop)" and loop_count > 0:
                    cv2.rectangle(annotated, (10, 70), (250, 120), (0, 0, 0), -1)
                    cv2.putText(annotated, f"Loop: {loop_count}", (20, 105),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                
                if pothole_detected:
                    alert_placeholder.markdown(
                        '<div class="alert-box">⚠️ POTHOLE AHEAD! ⚠️</div>',
                        unsafe_allow_html=True
                    )
                else:
                    alert_placeholder.empty()
                
                video_placeholder.image(annotated, channels="RGB", use_container_width=True)
                
                with col2:
                    fps_metric.metric("⚡ FPS", f"{fps:.1f}")
                    detected_metric.metric("🎯 Total", total_detected)
                    
                    if detections:
                        details_html = ""
                        for cls, count in detections.items():
                            info = CLASS_INFO.get(cls, (cls, '#999', False))
                            details_html += f"""
                            <div style="
                                background: {info[1]}22;
                                border-left: 4px solid {info[1]};
                                padding: 0.5rem;
                                margin: 0.5rem 0;
                                border-radius: 5px;
                            ">
                                <strong>{info[0]}</strong><br>
                                {count}x
                            </div>
                            """
                        detection_details.markdown(details_html, unsafe_allow_html=True)
                    else:
                        detection_details.info("ℹ️ No damage")
                
                time.sleep(0.01)
            
            cap.release()
            if uploaded_file:
                try:
                    os.unlink(video_path)
                except:
                    pass
            st.session_state.running = False

else:
    st.info("""
    ### 🎯 Cara Pakai:
    
    **Mode Live Demo (Recommended untuk presentasi):**
    1. Upload video dashcam sekali
    2. Klik "Start Detection"  
    3. Video akan loop otomatis seperti live stream
    4. Cocok untuk demo ke dosen!
    
    **Mode Upload Video:**
    - Upload video, play sekali, selesai
    
    **Mode Upload Image:**
    - Upload foto untuk analisis static
    
    💡 **Tip**: Pakai "Live Demo" mode dengan video 30-60 detik untuk demo smooth!
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>🎓 Road Damage Detection System</b></p>
    <p>YOLOv8 | Cloud-Compatible | Live Demo Mode</p>
    <p style="font-size: 0.85rem; margin-top: 0.5rem;">
        💡 Demo mode: Upload video → Auto-loop → Terlihat seperti real-time!
    </p>
</div>
""", unsafe_allow_html=True)
