"""
🚗 Road Damage Detection - WebRTC Real-time
Works on Cloud with browser camera access!
"""

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
from ultralytics import YOLO
import numpy as np
import av
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

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state.detections = {}
if 'total_detected' not in st.session_state:
    st.session_state.total_detected = 0
if 'last_alert' not in st.session_state:
    st.session_state.last_alert = 0
if 'fps' not in st.session_state:
    st.session_state.fps = 0

# Header
st.markdown('<h1 class="main-header">🛣️ Road Damage Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">📹 Real-time Browser Camera | Works on Cloud!</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
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
    
    st.markdown("---")
    st.markdown("""
    **💡 How it works:**
    - Click "START" below
    - Allow camera access in browser
    - Wait ~5-10 seconds for connection
    - Real-time detection starts!
    
    **🔧 Troubleshooting:**
    - Using Google STUN + Open Relay TURN
    - Works behind firewalls/NAT
    - If stuck, refresh page & try again
    """)

# Main content
st.markdown("### 📹 Live Camera Feed")
st.info("👇 Click START and allow camera access. Connection may take 5-10 seconds...")

# Alert placeholder
alert_placeholder = st.empty()

# Video callback class
class VideoProcessor:
    def __init__(self):
        self.confidence = confidence
        self.show_labels = show_labels
        self.show_conf = show_conf
        self.frame_count = 0
        self.start_time = time.time()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Run detection
        results = model.predict(
            img,
            conf=self.confidence,
            verbose=False,
            imgsz=640
        )
        
        # Annotate
        annotated = results[0].plot(
            labels=self.show_labels,
            conf=self.show_conf
        )
        
        # Count detections
        detections = {}
        pothole_detected = False
        
        for box in results[0].boxes:
            cls_name = model.names[int(box.cls[0])]
            detections[cls_name] = detections.get(cls_name, 0) + 1
            if cls_name == 'Pothole':
                pothole_detected = True
        
        # Update session state
        st.session_state.detections = detections
        st.session_state.total_detected += sum(detections.values())
        
        # Alert
        current_time = time.time()
        if pothole_detected and enable_audio and AUDIO_AVAILABLE:
            if current_time - st.session_state.last_alert > alert_cooldown:
                generate_beep()
                st.session_state.last_alert = current_time
        
        # Calculate FPS
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        st.session_state.fps = fps
        
        # Draw FPS
        cv2.rectangle(annotated, (10, 10), (200, 60), (0, 0, 0), -1)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# WebRTC configuration with STUN and TURN servers
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720}
        },
        "audio": False
    },
    async_processing=True,
)

# Statistics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("⚡ FPS", f"{st.session_state.fps:.1f}")

with col2:
    st.metric("🎯 Total Detected", st.session_state.total_detected)

with col3:
    if st.session_state.detections and 'Pothole' in st.session_state.detections:
        st.metric("🕳️ Potholes", st.session_state.detections['Pothole'])
    else:
        st.metric("🕳️ Potholes", 0)

# Show alert if pothole detected
if st.session_state.detections and 'Pothole' in st.session_state.detections:
    alert_placeholder.markdown(
        '<div class="alert-box">⚠️ POTHOLE DETECTED! ⚠️</div>',
        unsafe_allow_html=True
    )

# Detection details
st.markdown("### 🏷️ Current Detections")

if st.session_state.detections:
    cols = st.columns(len(st.session_state.detections))
    for idx, (cls, count) in enumerate(st.session_state.detections.items()):
        info = CLASS_INFO.get(cls, (cls, '#999', False))
        with cols[idx]:
            st.markdown(f"""
            <div style="
                background: {info[1]}22;
                border-left: 4px solid {info[1]};
                padding: 1rem;
                border-radius: 5px;
                text-align: center;
            ">
                <h3 style="margin: 0;">{info[0]}</h3>
                <p style="font-size: 2rem; margin: 0.5rem 0;">{count}</p>
            </div>
            """, unsafe_allow_html=True)
else:
    st.success("✅ No damage detected")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p><b>🎓 Road Damage Detection</b> | YOLOv8 + WebRTC</p>
    <p>📱 Works on Desktop & Mobile | ☁️ Deployed on Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
