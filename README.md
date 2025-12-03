# 🛣️ Real-time Road Damage Detection
**Deteksi kerusakan jalan secara real-time menggunakan YOLOv8 + Transfer Learning**
## 📋 Overview

Aplikasi ini menggunakan **Deep Learning** untuk mendeteksi kerusakan jalan secara otomatis dari video/webcam. Cocok digunakan sebagai sistem dashcam untuk membantu identifikasi kerusakan infrastruktur jalan.

### 🎯 Jenis Kerusakan yang Dideteksi

| Tipe | Deskripsi | Alert |
|------|-----------|-------|
| 📏 **Longitudinal Crack** | Retak memanjang searah jalan | - |
| ↔️ **Transverse Crack** | Retak melintang tegak lurus jalan | - |
| 🐊 **Alligator Crack** | Retak seperti kulit buaya (fatigue) | - |
| 🕳️ **Pothole** | Lubang pada permukaan jalan | 🔊 Audio Alert |

---

## ✨ Features

- 🎥 **Real-time Detection** - Deteksi langsung dari webcam/dashcam
- 🔊 **Audio Alert** - Peringatan suara saat mendeteksi pothole
- ⚡ **Optimized Performance** - Smooth 30+ FPS pada CPU
- 📊 **Live Statistics** - FPS, jumlah deteksi, dan detail real-time
- 🎨 **Modern UI** - Interface Streamlit yang clean dan responsive

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Webcam/Camera

### Setup

```bash
# Clone repository
git clone https://github.com/YabesEdward/RoadDamageDetection.git
cd RoadDamageDetection

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Requirements

```
streamlit
ultralytics
opencv-python
numpy
Pillow
pygame
```

---

## 💻 Usage

1. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

2. **Buka browser:** `http://localhost:8501`

3. **Konfigurasi:**
   - Pilih camera source
   - Atur confidence threshold
   - Enable/disable audio alert

4. **Start Detection:** Centang "Start Detection" untuk mulai

---

## 🔬 Training

Model di-training menggunakan **Transfer Learning** dengan YOLOv8 pada dataset RDD2022.

### Dataset
- **Source:** Road Damage Detection Challenge 2022
- **Countries:** Japan, India
- **Classes:** 4 (Longitudinal Crack, Transverse Crack, Alligator Crack, Pothole)

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8s |
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640x640 |
| Optimizer | Auto (AdamW) |

### Training Notebook
Lihat `RoadDamageDetection_Training_Colab.ipynb` untuk detail training di Google Colab.

---

## 📊 Results

### Model Performance

| Metric | Value |
|--------|-------|
| **mAP50** | 0.536 |
| **mAP50-95** | 0.242 |
| **Precision** | 0.588 |
| **Recall** | 0.505 |

### Confusion Matrix

Model menunjukkan performa terbaik pada:
- ✅ Alligator Crack (1125 correct)
- ✅ Pothole (607 correct)
- ✅ Longitudinal Crack (591 correct)

---

## 📁 Project Structure

```
RoadDamageDetection/
├── app.py                                    # Streamlit application
├── best.pt                                   # Trained YOLOv8 model
├── RoadDamageDetection_Training_Colab.ipynb  # Training notebook
├── requirements.txt                          # Dependencies
└── README.md                                 # Documentation
```

---

## 🛠️ Tech Stack

- **Deep Learning:** YOLOv8 (Ultralytics)
- **Framework:** PyTorch
- **Web App:** Streamlit
- **Computer Vision:** OpenCV
- **Audio:** Pygame, Winsound

---

## 📚 References

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [RDD2022 Dataset](https://github.com/sekilab/RoadDamageDetector)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

**🎓 Deep Learning Course Project**

Made with ❤️ using YOLOv8 + Streamlit

</div>
