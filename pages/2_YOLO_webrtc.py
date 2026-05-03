import threading
import time

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

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
    return YOLO_Pred(
        onnx_model=MODEL_PATH,
        data_yaml=DATA_PATH,
    )


yolo = load_yolo_model()


rtc_configuration = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)


class FastYOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()

        self.frame_count = 0
        self.process_every_n_frames = 5

        self.latest_phone_count = 0
        self.latest_boxes_frame = None
        self.last_inference_time = 0.0

        self.target_width = 416
        self.target_height = 312

    def recv(self, frame):
        start_time = time.time()

        img = frame.to_ndarray(format="bgr24")

        img = cv2.resize(
            img,
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_AREA,
        )

        output_img = img.copy()
        self.frame_count += 1

        try:
            should_process = self.frame_count % self.process_every_n_frames == 0

            if should_process:
                pred_img, phone_count = yolo.predictions(img.copy())

                inference_ms = (time.time() - start_time) * 1000

                with self.lock:
                    self.latest_phone_count = phone_count
                    self.latest_boxes_frame = pred_img.copy()
                    self.last_inference_time = inference_ms

                output_img = pred_img

            else:
                with self.lock:
                    phone_count = self.latest_phone_count
                    inference_ms = self.last_inference_time

                    if self.latest_boxes_frame is not None:
                        output_img = self.latest_boxes_frame.copy()
                    else:
                        output_img = img.copy()

            cv2.putText(
                output_img,
                f"Phones: {phone_count}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                output_img,
                f"Inference: {inference_ms:.0f} ms",
                (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(output_img, format="bgr24")

        except Exception as e:
            print(f"YOLO/WebRTC error: {e}")

            cv2.putText(
                output_img,
                "Detection error",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(output_img, format="bgr24")


st.title("Fast YOLO Phone Detection")

st.warning(
    "For best speed, use low resolution and low FPS. "
    "Streamlit Cloud runs this on CPU, so real-time YOLO can still be slow."
)

webrtc_streamer(
    key="fast-yolo-phone-detection",
    video_processor_factory=FastYOLOVideoProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 416, "max": 416},
            "height": {"ideal": 312, "max": 312},
            "frameRate": {"ideal": 8, "max": 10},
        },
        "audio": False,
    },
    async_processing=True,
)
