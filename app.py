import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Deteksi Kerusakan Jalan",
    page_icon="🛣️",
    layout="wide"
)

# Load model with caching
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

CLASS_INFO = {
    'Longitudinal_Crack': '🔹 Retak Memanjang',
    'Transverse_Crack': '↔️ Retak Melintang',
    'Alligator_Crack': '🐊 Retak Buaya',
    'Pothole': '🕳️ Lubang'
}

# Header
st.title("🛣️ Deteksi Kerusakan Jalan")
st.markdown("📸 Upload gambar jalan untuk mendeteksi kerusakan seperti retak dan lubang")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
conf = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Tingkat kepercayaan deteksi (semakin tinggi semakin akurat)"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 Cara Penggunaan:
1. Upload gambar jalan
2. Atur confidence threshold
3. Lihat hasil deteksi

### 🏷️ Jenis Kerusakan:
- 🔹 Retak Memanjang
- ↔️ Retak Melintang
- 🐊 Retak Buaya
- 🕳️ Lubang
""")

# File uploader
uploaded_file = st.file_uploader(
    "Upload Gambar Jalan",
    type=["jpg", "jpeg", "png"],
    help="Format: JPG, JPEG, atau PNG"
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Gambar Asli")
        st.image(image, use_container_width=True)
    
    # Predict with progress
    with st.spinner("🔍 Sedang mendeteksi kerusakan..."):
        results = model.predict(image_np, conf=conf, verbose=False, imgsz=640)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        # Count detections
        detections = {}
        for box in results[0].boxes:
            cls = model.names[int(box.cls[0])]
            detections[cls] = detections.get(cls, 0) + 1
    
    # Display annotated image
    with col2:
        st.subheader("🎯 Hasil Deteksi")
        st.image(annotated, use_container_width=True)
    
    # Display summary
    st.markdown("---")
    st.subheader("📊 Ringkasan Deteksi")
    
    if not detections:
        st.success("✅ Tidak ada kerusakan jalan terdeteksi")
    else:
        # Total detections
        total = sum(detections.values())
        st.info(f"🎯 **Total Deteksi: {total} kerusakan**")
        
        # Display each detection type
        cols = st.columns(len(detections))
        for idx, (cls, count) in enumerate(detections.items()):
            with cols[idx]:
                st.metric(
                    label=CLASS_INFO.get(cls, cls),
                    value=f"{count}x"
                )
        
        # Detailed table
        st.markdown("### 📋 Detail Kerusakan:")
        for cls, count in detections.items():
            st.write(f"- {CLASS_INFO.get(cls, cls)}: **{count} titik**")

else:
    # Welcome message
    st.info("👆 Silakan upload gambar jalan untuk memulai deteksi kerusakan")
    
    # Example section
    st.markdown("---")
    st.markdown("""
    ### 💡 Tips Penggunaan:
    - Gunakan gambar dengan resolusi yang baik
    - Pastikan area jalan terlihat jelas
    - Atur confidence threshold sesuai kebutuhan:
      - **0.1 - 0.3**: Deteksi lebih sensitif (lebih banyak hasil)
      - **0.3 - 0.5**: Deteksi seimbang (rekomendasi)
      - **0.5 - 1.0**: Deteksi sangat ketat (hasil lebih akurat)
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center'>Made with ❤️ using Streamlit & YOLOv8</div>",
    unsafe_allow_html=True
)
