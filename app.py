"""
🚗 Road Damage Detection - Cloud Compatible
Supports: Upload Video, Upload Image, or Local Webcam
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time
import tempfile

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
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

# Generate simple beep sound
def generate_beep():
    """Generate a simple beep sound"""
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

# Header
st.markdown('<h1 class="main-header">🛣️ Road Damage Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📤 Upload Video/Image | 📹 Local Webcam | 🔊 Audio Alerts</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Input source selection
    st.markdown("### 📥 Input Source")
    input_mode = st.radio(
        "Select Input",
        ["📤 Upload Video", "🖼️ Upload Image", "📹 Webcam (Local Only)"],
        help="Webcam only works when running locally"
    )
    
    uploaded_file = None
    camera_idx = 0
    
    if input_mode == "📤 Upload Video":
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload dashcam video for analysis"
        )
    elif input_mode == "🖼️ Upload Image":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload road image for analysis"
        )
    else:  # Webcam
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
    enable_audio = st.checkbox("Enable Audio Alert", value=AUDIO_AVAILABLE)
    if not AUDIO_AVAILABLE:
        st.warning("⚠️ Audio not available")
    
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
    - Upload video/image works on Cloud
    - Webcam only works locally
    - Supported formats: MP4, AVI, MOV, JPG, PNG
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

# Process input
if input_mode == "🖼️ Upload Image" and uploaded_file:
    # Process single image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:  # Grayscale
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    # Detection
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
    
    # Display
    if pothole_detected:
        alert_placeholder.markdown(
            '<div class="alert-box">⚠️ POTHOLE DETECTED! ⚠️</div>',
            unsafe_allow_html=True
        )
        if enable_audio and AUDIO_AVAILABLE:
            generate_beep()
    
    video_placeholder.image(annotated, channels="RGB", use_container_width=True)
    
    # Metrics
    with col2:
        detected_metric.metric("🎯 Total Detected", sum(detections.values()))
        
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
            detection_details.success("✅ No damage detected")

elif (input_mode == "📤 Upload Video" and uploaded_file) or input_mode == "📹 Webcam (Local Only)":
    # Process video or webcam
    run_btn = st.button("▶️ Start Detection", use_container_width=True)
    stop_btn = st.button("⏹️ Stop Detection", use_container_width=True)
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if run_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False
    
    if st.session_state.running:
        # Open video source
        if input_mode == "📤 Upload Video":
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
        else:  # Webcam
            cap = cv2.VideoCapture(camera_idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            st.error("❌ Cannot open video source. If using webcam, make sure you're running locally.")
        else:
            st.success("✅ Video source active!")
            
            frame_count = 0
            start_time = time.time()
            total_detected = 0
            last_alert_time = 0
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.warning("⏹️ Video ended")
                    break
                
                # Detection
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
                
                # Alert
                current_time = time.time()
                if pothole_detected and current_time - last_alert_time > alert_cooldown:
                    if enable_audio and AUDIO_AVAILABLE:
                        generate_beep()
                    last_alert_time = current_time
                
                # FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                if show_fps:
                    cv2.rectangle(annotated, (10, 10), (200, 60), (0, 0, 0), -1)
                    cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                # Visual alert
                if pothole_detected:
                    alert_placeholder.markdown(
                        '<div class="alert-box">⚠️ POTHOLE AHEAD! ⚠️</div>',
                        unsafe_allow_html=True
                    )
                else:
                    alert_placeholder.empty()
                
                # Display
                video_placeholder.image(annotated, channels="RGB", use_container_width=True)
                
                # Metrics
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
            st.session_state.running = False
else:
    st.info("👆 Please upload a file or select webcam to begin")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "🎓 <b>Road Damage Detection</b> | YOLOv8 | Upload or Webcam Support"
    "</p>",
    unsafe_allow_html=True
)
