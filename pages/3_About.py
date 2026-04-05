import streamlit as st
from ctdiheader import ctdiheader

st.set_page_config(
    page_title="About",
    layout="wide"
)

# Header
ctdiheader()

st.markdown('<div class="content">', unsafe_allow_html=True)

st.title("About This Application")

st.markdown("""
This application is a **YOLO-based Object Detection System** designed to identify mobile phones from images and live camera input.

It demonstrates how modern computer vision models can be integrated into a simple and interactive web interface using Streamlit.

### Key Highlights
- Real-time object detection using YOLO
- Image and webcam-based input support
- Fast and efficient processing
- Clean and user-friendly interface

### Use Case
This project can be applied in environments such as **device testing, quality control, and automation workflows**, similar to real-world industrial setups.

### Technologies Used
- Python
- OpenCV
- YOLO (ONNX model)
- Streamlit

---

This project showcases practical implementation of AI in a web-based environment and serves as a foundation for building more advanced computer vision applications.
""")

st.markdown('</div>', unsafe_allow_html=True)