import streamlit as st
from ctdiheader import ctdiheader

st.set_page_config(
    page_title="Home",
    layout="wide"
)

# Header
ctdiheader()

# Content container
st.markdown('<div class="content">', unsafe_allow_html=True)

st.title("YOLO V5 Object Detection App")
st.caption("This web application demonstrates object detection")

st.markdown("""
### Features
- Detects phones from images
- Fast YOLO-based detection
- Scalable with high potential for future enhancements

### Navigation
Use the sidebar to open detection pages.
""")

st.page_link("pages/1_YOLO_for_image.py", label="Open YOLO for Image", icon="📷")

st.markdown('</div>', unsafe_allow_html=True)
