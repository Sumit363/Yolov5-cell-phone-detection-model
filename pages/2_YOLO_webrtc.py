import threading

import av
import cv2
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)

from ctdiheader import ctdiheader
from yolo_predictions import YOLO_Pred


st.set_page_config(
    page_title="YOLO Phone Detection",
    layout="wide",
)

ctdiheader()


MODEL_PATH = "./models/best.onnx"
DATA_PATH = "./models/data.yaml"


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_yolo_model():
    return YOLO_Pred(MODEL_PATH, DATA_PATH)


yolo = load_yolo_model()


rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.frame_count = 0
        self.process_every_n_frames = 2
        self.latest_phone_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        output_img = img.copy()

        self.frame_count += 1

        try:
            should_process = self.frame_count % self.process_every_n_frames == 0

            if should_process:
                pred_img, phone_count = yolo.predictions(img)

                with self.lock:
                    self.latest_phone_count = phone_count

                output_img = pred_img
            else:
                with self.lock:
                    phone_count = self.latest_phone_count

                output_img = img.copy()

            cv2.putText(
                output_img,
                f"Phones: {phone_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(output_img, format="bgr24")

        except Exception as e:
            print(f"YOLO/WebRTC error: {e}")

            cv2.putText(
                output_img,
                "Detection error",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(output_img, format="bgr24")


st.title("YOLO Phone Detection")

st.info(
    "Camera detection runs in the browser using WebRTC. "
    "If this works on home Wi-Fi or mobile data but fails at work, "
    "your workplace network may be blocking WebRTC traffic."
)

webrtc_streamer(
    key="yolo-phone-detection",
    video_processor_factory=YOLOVideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "frameRate": {"ideal": 15, "max": 20},
        },
        "audio": False,
    },
    async_processing=True,
)
