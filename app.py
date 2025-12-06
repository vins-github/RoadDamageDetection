"""
🚗 Road Damage Detection - Gradio Version
✅ Support Webcam Real-time
✅ Deploy ke Hugging Face Spaces GRATIS
"""

import gradio as gr
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import time

# Load model
model = YOLO('best.pt')

# Class info dengan emoji
CLASS_INFO = {
    'Longitudinal_Crack': '🔹 Retak Memanjang',
    'Transverse_Crack': '↔️ Retak Melintang',
    'Alligator_Crack': '🐊 Retak Buaya',
    'Pothole': '🕳️ Lubang'
}

# Detection function for image/webcam
def detect_damage(image, confidence=0.25):
    """
    Detect road damage from image or webcam frame
    """
    if image is None:
        return None, "⚠️ No image provided"
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Run YOLO detection
    results = model.predict(img_array, conf=confidence, verbose=False, imgsz=640)
    
    # Get annotated image
    annotated = results[0].plot(labels=True, conf=True)
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    
    # Count detections
    detections = {}
    pothole_detected = False
    
    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])]
        detections[cls_name] = detections.get(cls_name, 0) + 1
        if cls_name == 'Pothole':
            pothole_detected = True
    
    # Build result text
    if not detections:
        result_text = "✅ **No damage detected!**\n\nRoad condition looks good."
    else:
        result_text = f"### 🎯 Detection Results\n\n"
        result_text += f"**Total Detections: {sum(detections.values())}**\n\n"
        
        if pothole_detected:
            result_text += "### ⚠️ **POTHOLE ALERT!** ⚠️\n\n"
        
        result_text += "**Details:**\n"
        for cls, count in detections.items():
            emoji_name = CLASS_INFO.get(cls, cls)
            result_text += f"- {emoji_name}: **{count}x**\n"
    
    return annotated, result_text

# Detection function for video
def detect_video(video_path, confidence=0.25, skip_frames=2):
    """
    Process video file and detect road damage
    """
    if video_path is None:
        return None, "⚠️ No video provided"
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None, "❌ Cannot open video file"
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video
    output_path = "output_detected.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_detections = 0
    pothole_frames = 0
    
    status_text = f"Processing video... {total_frames} frames\n\n"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every N frames
        if frame_count % skip_frames == 0:
            results = model.predict(frame, conf=confidence, verbose=False, imgsz=640)
            annotated = results[0].plot(labels=True, conf=True)
            
            # Count detections
            detections = len(results[0].boxes)
            total_detections += detections
            
            # Check for potholes
            for box in results[0].boxes:
                if model.names[int(box.cls[0])] == 'Pothole':
                    pothole_frames += 1
                    break
            
            out.write(annotated)
        else:
            out.write(frame)
    
    cap.release()
    out.release()
    
    # Build result text
    result_text = f"""
### ✅ Video Processing Complete!

**Statistics:**
- Total Frames: {total_frames}
- Frames Processed: {frame_count // skip_frames}
- Total Detections: {total_detections}
- Frames with Potholes: {pothole_frames}

**Output:** Saved as `output_detected.mp4`
"""
    
    return output_path, result_text

# Custom CSS
custom_css = """
#warning {
    background: linear-gradient(90deg, #ff4444, #cc0000);
    color: white;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 1.5em;
    font-weight: bold;
    margin: 10px 0;
}
.gradio-container {
    font-family: 'Segoe UI', Arial, sans-serif;
}
"""

# Build Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🛣️ Road Damage Detection System
    ### 📹 Real-time Detection using YOLOv8 | Support Webcam, Upload, & Video
    """)
    
    with gr.Tabs():
        
        # Tab 1: Webcam Real-time
        with gr.Tab("📹 Real-time Webcam"):
            gr.Markdown("""
            ### 🎥 Use your webcam for real-time road damage detection
            **Instructions:**
            1. Click "Start Recording" to activate webcam
            2. Point camera at road surface
            3. Detection will run automatically
            4. Adjust confidence threshold as needed
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    webcam_input = gr.Image(
                        sources=["webcam"],
                        type="pil",
                        label="📹 Webcam Feed",
                        streaming=True
                    )
                    
                with gr.Column(scale=1):
                    webcam_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="🎯 Confidence Threshold"
                    )
                    webcam_output_text = gr.Markdown(label="📊 Detection Results")
            
            with gr.Row():
                webcam_output_img = gr.Image(label="🎯 Detected Image", type="numpy")
            
            # Auto-detect on webcam frame change
            webcam_input.stream(
                fn=detect_damage,
                inputs=[webcam_input, webcam_confidence],
                outputs=[webcam_output_img, webcam_output_text],
                stream_every=0.5  # Process every 0.5 seconds
            )
        
        # Tab 2: Upload Image
        with gr.Tab("🖼️ Upload Image"):
            gr.Markdown("""
            ### 📤 Upload a photo of road damage
            **Supported formats:** JPG, PNG, JPEG
            """)
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        type="pil",
                        label="📤 Upload Road Image"
                    )
                    image_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="🎯 Confidence Threshold"
                    )
                    image_button = gr.Button("🔍 Detect Damage", variant="primary")
                
                with gr.Column():
                    image_output = gr.Image(label="🎯 Detection Result", type="numpy")
                    image_output_text = gr.Markdown(label="📊 Analysis")
            
            image_button.click(
                fn=detect_damage,
                inputs=[image_input, image_confidence],
                outputs=[image_output, image_output_text]
            )
            
            # Example images
            gr.Examples(
                examples=[
                    ["examples/pothole.jpg", 0.25],
                    ["examples/crack.jpg", 0.3],
                ],
                inputs=[image_input, image_confidence],
                outputs=[image_output, image_output_text],
                fn=detect_damage,
                cache_examples=False
            )
        
        # Tab 3: Upload Video
        with gr.Tab("🎬 Upload Video"):
            gr.Markdown("""
            ### 📹 Process dashcam or recorded video
            **Supported formats:** MP4, AVI, MOV
            **Note:** Processing may take time depending on video length
            """)
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(label="📤 Upload Video")
                    video_confidence = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="🎯 Confidence Threshold"
                    )
                    video_skip = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=2,
                        step=1,
                        label="⚡ Skip Frames (higher = faster)"
                    )
                    video_button = gr.Button("🎬 Process Video", variant="primary")
                
                with gr.Column():
                    video_output = gr.Video(label="🎯 Processed Video")
                    video_output_text = gr.Markdown(label="📊 Statistics")
            
            video_button.click(
                fn=detect_video,
                inputs=[video_input, video_confidence, video_skip],
                outputs=[video_output, video_output_text]
            )
    
    # Footer
    gr.Markdown("""
    ---
    ### 💡 Tips for Best Results:
    - **Webcam**: Ensure good lighting and stable camera position
    - **Images**: Use clear, high-resolution photos
    - **Videos**: Dashcam footage works best at 720p or 1080p
    - **Confidence**: Lower threshold (0.15-0.25) for more detections, higher (0.4-0.6) for more accurate
    
    ### 📊 Detection Classes:
    - 🔹 **Longitudinal Crack** - Retak memanjang
    - ↔️ **Transverse Crack** - Retak melintang  
    - 🐊 **Alligator Crack** - Retak buaya
    - 🕳️ **Pothole** - Lubang jalan
    
    ---
    **🎓 Developed for Deep Learning & Computer Vision Course**
    
    *Powered by YOLOv8 + Gradio*
    """)

# Launch app
if __name__ == "__main__":
    demo.launch(
        share=True,  # Create public link
        server_name="0.0.0.0",
        server_port=7860
    )
