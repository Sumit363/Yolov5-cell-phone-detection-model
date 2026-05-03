import streamlit as st

st.set_page_config(
    page_title="YOLO Object Detection",
    layout="wide",
    page_icon="./images/object.png",
)

from PIL import Image
import numpy as np

from ctdiheader import ctdiheader
from yolo_predictions import YOLO_Pred


MODEL_PATH = "./models/best.onnx"
DATA_PATH = "./models/data.yaml"


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo_model():
    return YOLO_Pred(
        onnx_model=MODEL_PATH,
        data_yaml=DATA_PATH,
    )


def upload_image():
    image_file = st.file_uploader(
        label="Upload Image",
        type=["png", "jpg", "jpeg"],
    )

    if image_file is None:
        return None

    size_mb = image_file.size / (1024**2)

    file_details = {
        "filename": image_file.name,
        "filetype": image_file.type,
        "filesize": f"{size_mb:,.2f} MB",
    }

    return {
        "file": image_file,
        "details": file_details,
    }


def main():
    ctdiheader()

    st.header("YOLO Object Detection for Images")
    st.write("Upload an image to detect phones using your YOLO model.")

    yolo = load_yolo_model()

    uploaded = upload_image()

    if uploaded is None:
        st.info("Please upload a PNG, JPG, or JPEG image.")
        return

    image_obj = Image.open(uploaded["file"]).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_obj, use_container_width=True)

    with col2:
        st.subheader("File Details")
        st.json(uploaded["details"])

        detect_button = st.button("Get Detection from YOLO", type="primary")

    if detect_button:
        with st.spinner("Detecting objects. Please wait..."):
            image_array = np.array(image_obj)

            result = yolo.predictions(image_array)

            # Your YOLO_Pred now returns: image, phone_count
            if isinstance(result, tuple):
                pred_img, phone_count = result
            else:
                pred_img = result
                phone_count = None

            pred_img_obj = Image.fromarray(pred_img)

        st.subheader("Predicted Image")
        st.caption("Object detection result from YOLOv5 model")

        if phone_count is not None:
            st.success(f"Phones detected: {phone_count}")

        st.image(pred_img_obj, use_container_width=True)


if __name__ == "__main__":
    main()
