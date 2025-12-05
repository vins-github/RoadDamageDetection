"""
🚗 Road Damage Detection - Hybrid Mode
Cloud: Upload Video/Image | Local: Real-time Webcam
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import tempfile
import os

# Try to initialize pygame (optional for audio)
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
    .info-card {
        background: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
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

model = load_model()

# Class info
CLASS_INFO = {
    'Longitudinal_Crack': ('🔹 Retak Memanjang', '#FF6B6B', False),
    'Transverse_Crack': ('↔️ Retak Melintang', '#4ECDC4', False),
    'Alligator_Crack': ('🐊 Retak Buaya', '#95E1D3', False),
    'Pothole': ('🕳️ Lubang', '#FFE66D', True)
}

# Detect if running locally
def is_local():
    """Check if running on local machine"""
    try:
        cap = cv2.VideoCapture(0)
        ret = cap.isOpened()
        cap.release()
        return ret
    except:
        return False

RUNNING_LOCAL = is_local()

# Header
st.markdown('<h1 class="main-header">🛣️ Road Damage Detection System</h1>', unsafe_allow_html=True)

if RUNNING_LOCAL:
    st.markdown('<p class="subtitle">💻 Local Mode: Real-time Webcam Available!</p>', unsafe_allow_html=True)
else:
    st.markdown('<p class="subtitle">☁️ Cloud Mode: Upload Video/Image</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Mode detection
    if RUNNING_LOCAL:
        st.success("✅ Running Locally - Webcam Available")
    else:
        st.info("☁️ Running on Cloud - Upload Mode")
    
    st.markdown("---")
    
    # Input source selection
    st.markdown("### 📥 Input Source")
    
    if RUNNING_LOCAL:
        input_options = ["📹 Real-time Webcam", "📤 Upload Video", "🖼️ Upload Image"]
    else:
        input_options = ["📤 Upload Video", "🖼️ Upload Image"]
    
    input_mode = st.radio("Select Input", input_options)
    
    uploaded_file = None
    camera_idx = 0
    
    if input_mode == "📤 Upload Video":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
            help="Upload dashcam video for analysis"
        )
    elif input_mode == "🖼️ Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload road image for analysis"
        )
    elif input_mode == "📹 Real-time Webcam":
        camera_idx = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=0,
            help="Usually 0 for default camera"
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
    
    st.markdown("---")
    st.markdown("""
    **💡 Tips:**
    - **Local**: Use real-time webcam
    - **Cloud**: Upload video/image
    - Supported: MP4, AVI, MOV, JPG, PNG
    """)

# Info card about deployment
with st.expander("ℹ️ Untuk Dosen - Cara Demo Real-time", expanded=False):
    st.markdown("""
    ### 🎓 Cara Menunjukkan Real-time Detection:
    
    **Opsi 1: Demo Local (Recommended)**
    1. Jalankan aplikasi di laptop: `streamlit run app.py`
    2. Pilih "Real-time Webcam"
    3. Tunjukkan deteksi langsung dengan webcam/dashcam
    
    **Opsi 2: Cloud dengan Video Upload**
    1. Deploy ke Streamlit Cloud
    2. Record video dashcam terlebih dahulu
    3. Upload dan tunjukkan hasil deteksi
    
    **Opsi 3: Screen Recording**
    1. Record screen saat run local mode
    2. Tunjukkan video recording sebagai bukti
    3. Bisa di-upload ke cloud juga
    
    ### 📌 Catatan:
    - Webcam **HANYA bisa di local**, tidak di cloud
    - Cloud butuh upload karena server tidak punya kamera
    - Ini **batasan teknologi web**, bukan bug aplikasi
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
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
if input_mode == "🖼️ Upload Image" and uploaded_file:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    with st.spinner("🔍 Analyzing image..."):
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

# Process video or webcam
elif (input_mode == "📤 Upload Video" and uploaded_file) or input_mode == "📹 Real-time Webcam":
    
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
        if input_mode == "📤 Upload Video":
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:
            cap = cv2.VideoCapture(camera_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
        
        if not cap.isOpened():
            st.error("❌ Cannot open video source!")
        else:
            st.success("✅ Video source active!")
            
            frame_count = 0
            start_time = time.time()
            total_detected = 0
            last_alert_time = 0
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    if input_mode == "📤 Upload Video":
                        # Loop video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
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
            if input_mode == "📤 Upload Video":
                try:
                    os.unlink(tfile.name)
                except:
                    pass
            st.session_state.running = False
            st.warning("⏹️ Detection stopped")

else:
    # Show instructions
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    if RUNNING_LOCAL:
        st.markdown("""
        ### 🎯 Ready to Detect!
        
        **For Real-time Detection:**
        1. Select "📹 Real-time Webcam" in sidebar
        2. Click "▶️ Start Detection"
        3. Point camera at road/dashcam footage
        
        **For Video Analysis:**
        1. Select "📤 Upload Video"
        2. Choose your dashcam recording
        3. Click "▶️ Start Detection"
        """)
    else:
        st.markdown("""
        ### 📤 Upload Mode Active
        
        **Steps:**
        1. Select input type in sidebar
        2. Upload your video or image
        3. Click "▶️ Start Detection"
        
        **Note:** Real-time webcam only works when running locally.
        For demo purposes, you can:
        - Record a video on local machine
        - Upload it here for cloud demo
        """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>🎓 Road Damage Detection System</b></p>
    <p>YOLOv8 | Real-time & Upload Support | Local & Cloud Compatible</p>
</div>
""", unsafe_allow_html=True)
