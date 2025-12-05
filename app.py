"""
🚗 Road Damage Detection - Fixed for Streamlit Cloud
Audio-safe version with visual alerts
"""

import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

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

@st.cache_data
def detect_cameras():
    """Detect available cameras"""
    cameras = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                backend = cap.getBackendName()
                cameras.append((i, f"Camera {i} ({backend})"))
            cap.release()
    return cameras if cameras else [(0, "Camera 0 (Default)")]

# Generate simple beep sound
def generate_beep():
    """Generate a simple beep sound (frequency-based)"""
    if not AUDIO_AVAILABLE:
        return
    
    try:
        # Create a simple 440Hz beep for 0.2 seconds
        sample_rate = 22050
        duration = 0.2
        frequency = 440
        
        samples = int(sample_rate * duration)
        wave = np.sin(2 * np.pi * frequency * np.linspace(0, duration, samples))
        wave = (wave * 32767).astype(np.int16)
        
        # Convert to stereo
        stereo_wave = np.column_stack((wave, wave))
        
        sound = pygame.sndarray.make_sound(stereo_wave)
        sound.play()
    except Exception as e:
        pass  # Silently fail if audio doesn't work

model = load_model()
available_cameras = detect_cameras()

# Class info
CLASS_INFO = {
    'Longitudinal_Crack': ('🔹 Retak Memanjang', '#FF6B6B', False),
    'Transverse_Crack': ('↔️ Retak Melintang', '#4ECDC4', False),
    'Alligator_Crack': ('🐊 Retak Buaya', '#95E1D3', False),
    'Pothole': ('🕳️ Lubang', '#FFE66D', True)
}

# Header
st.markdown('<h1 class="main-header">🛣️ Road Damage Detection System</h1>', unsafe_allow_html=True)
audio_status = "🔊 Audio Available" if AUDIO_AVAILABLE else "🔇 Visual Alerts Only"
st.markdown(f'<p class="subtitle">🚗 Real-time Dashcam | {audio_status}</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Camera selection
    st.markdown("### 📹 Camera Source")
    
    camera_mode = st.radio(
        "Selection Mode",
        ["Auto-detect", "Manual Index"],
        horizontal=True
    )
    
    if camera_mode == "Auto-detect":
        camera_options = {desc: idx for idx, desc in available_cameras}
        selected_camera_desc = st.selectbox(
            "Select Camera",
            options=list(camera_options.keys())
        )
        camera_idx = camera_options[selected_camera_desc]
    else:
        camera_idx = st.number_input(
            "Camera Index",
            min_value=0,
            max_value=10,
            value=1
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
        st.warning("⚠️ Audio not available - using visual alerts only")
    
    alert_cooldown = st.slider(
        "Alert Cooldown (seconds)",
        min_value=1,
        max_value=10,
        value=3,
        help="Jarak minimal antar alert"
    )
    
    # Display options
    st.markdown("### 🎨 Display")
    show_labels = st.checkbox("Show Labels", value=True)
    show_conf = st.checkbox("Show Confidence", value=True)
    show_fps = st.checkbox("Show FPS", value=True)
    
    # Control
    st.markdown("### 🎮 Control")
    run_detection = st.checkbox("▶️ Start Detection", value=False)
    
    st.markdown("---")
    st.markdown("""
    **⚡ Features:**
    - Real-time detection
    - Visual alerts
    - Smooth video streaming
    - Cloud-compatible
    """)

# Main content
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📹 Live Video Feed")
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Statistics")
    fps_metric = st.empty()
    detected_metric = st.empty()
    
    st.markdown("### 🏷️ Detections")
    detection_details = st.empty()

# Video streaming
if run_detection:
    cap = cv2.VideoCapture(camera_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        st.error(f"❌ Cannot open camera {camera_idx}")
    else:
        st.success("✅ Camera active!")
        
        frame_count = 0
        start_time = time.time()
        total_detected = 0
        last_alert_time = 0
        
        while run_detection:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detection
            results = model.predict(
                frame,
                conf=confidence,
                verbose=False,
                device='cpu',
                imgsz=640
            )
            
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
            
            # Alert system
            current_time = time.time()
            if pothole_detected:
                if current_time - last_alert_time > alert_cooldown:
                    # Try audio alert
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
                alert_emoji = "🔊" if (enable_audio and AUDIO_AVAILABLE) else "⚠️"
                alert_placeholder.markdown(
                    f'<div class="alert-box">{alert_emoji} POTHOLE AHEAD! {alert_emoji}</div>',
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
                        alert_emoji = " 🔊" if (cls == 'Pothole' and AUDIO_AVAILABLE) else ""
                        details_html += f"""
                        <div style="
                            background: {info[1]}22;
                            border-left: 4px solid {info[1]};
                            padding: 0.5rem;
                            margin: 0.5rem 0;
                            border-radius: 5px;
                        ">
                            <strong>{info[0]}{alert_emoji}</strong><br>
                            {count}x
                        </div>
                        """
                    detection_details.markdown(details_html, unsafe_allow_html=True)
                else:
                    detection_details.info("ℹ️ No damage")
            
            time.sleep(0.01)
        
        cap.release()
        st.warning("⏹️ Stopped")
else:
    st.info("👆 Enable 'Start Detection' to begin")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>"
    "🎓 <b>Road Damage Detection</b> | YOLOv8 | Cloud-Compatible"
    "</p>",
    unsafe_allow_html=True
)
