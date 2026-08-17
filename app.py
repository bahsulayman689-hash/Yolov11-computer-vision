import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="YOLOv11 Vision Studio", page_icon="🎯", layout="wide")

# ==========================================
# SIDEBAR - DEVELOPER IDENTITY
# ==========================================
st.sidebar.title("🎯 Vision Workspace")
st.sidebar.markdown("### Production Setup")

with st.sidebar.container(border=True):
    st.markdown("**Role:** Deep & ML Engineer")
    st.caption("State-of-the-Art Real-Time Object Detection via YOLOv11 Nano Architecture.")
    st.markdown("""
    **Core Infrastructure Stack:**
    * 🧠 `Ultralytics YOLOv11`
    * 🎨 `Streamlit Framework`
    * 👁️ `OpenCV / PIL`
    """)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Developer Tip:** The Nano model is optimized to execute inference under 100ms on a standard CPU node.")

# ==========================================
# MAIN APPLICATION INTERFACE
# ==========================================
st.title("🎯 Real-Time AI Object Detection Engine")
st.markdown("Upload raw images to parse spatial visual features and extract labeled bounding boxes automatically.")


# Cached Model Loading: Downloads the official 5.4MB weight file instantly on first launch
@st.cache_resource
def load_yolo_model():
    # Will automatically fetch yolo11n.pt (5.4 MB) onto the host server memory
    model = YOLO("yolo11n.pt")
    return model


with st.spinner("Initializing YOLOv11 Vision Framework layers..."):
    model = load_yolo_model()

# Threshold adjustments inside the UI layout
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    conf_threshold = st.slider("Confidence Threshold Score:", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
with col_ctrl2:
    st.caption("Lowering the score shows more potential detections, while increasing it filters out errors.")

st.divider()

# Core App Input & Output Panel Split
col_input, col_output = st.columns(2)

with col_input:
    st.subheader("🖼️ 1. Source Image Upload")
    uploaded_file = st.file_uploader("Choose a JPG, JPEG, or PNG image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Original Uploaded Target Asset", use_container_width=True)

with col_output:
    st.subheader("🔍 2. AI Analytical Detection View")

    if uploaded_file is not None:
        if st.button("✨ Run YOLOv11 Detection Pass", type="primary", use_container_width=True):
            with st.spinner("Executing forward pass predictions through neural layers..."):
                # Convert PIL Image back to a standard numpy matrix array structure
                image_array = np.array(input_image)

                # Execute bounding-box inference
                results = model.predict(source=image_array, conf=conf_threshold, verbose=False)

                # Plot/draw the output labels and boxes using Ultralytics utility extensions
                annotated_frame = results[0].plot()

                # Convert the raw array representation back to a visual PIL interface component
                output_image = Image.fromarray(annotated_frame)
                st.image(output_image, caption="YOLOv11 Processed Spatial Mapping Output", use_container_width=True)

                # Display structural inventory telemetry text logs
                st.success("🎉 Computer Vision inference execution complete!")

                # Extract detected unique classes found inside the scene frame bounding matrix
                boxes = results[0].boxes
                if len(boxes) > 0:
                    st.markdown("### 📊 Extracted Target Class Inventory Logs:")
                    detected_names = []
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = model.names[class_id]
                        confidence = float(box.conf[0])
                        detected_names.append(f"- **{class_name.upper()}** (Certainty: {confidence * 100:.1f}%)")

                    st.markdown("\n".join(detected_names))
                else:
                    st.warning("No structural object bounding targets cleared your Confidence Threshold setting.")
    else:
        st.info("Please upload a picture in the left panel to engage the YOLO neural networks pipeline.")
