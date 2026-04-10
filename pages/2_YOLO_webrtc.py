import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from yolo_predictions import YOLO_Pred
from ctdiheader import ctdiheader

ctdiheader()

# load yolo model
yolo = YOLO_Pred('./models/best.onnx', './models/data.yaml')


def video_frame_callback(frame):
    try:
        img = frame.to_ndarray(format="bgr24")
        pred_img, phone_count = yolo.predictions(img)

        cv2.putText(
            pred_img,
            f"Phones: {phone_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

    except Exception as e:
        print("Error:", e)
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="example",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False}
)