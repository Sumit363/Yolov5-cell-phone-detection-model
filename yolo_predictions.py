#!/usr/bin/env python
# coding: utf-8

import os

import cv2
import numpy as np
import onnxruntime as ort
import yaml
from yaml.loader import SafeLoader


class YOLO_Pred:
    def __init__(self, onnx_model, data_yaml):
        self.onnx_model = onnx_model
        self.data_yaml = data_yaml
        self.input_size = 640
        self.conf_threshold = 0.4
        self.score_threshold = 0.25
        self.nms_threshold = 0.45

        self._validate_files()

        self.labels = self._load_labels()
        self.nc = len(self.labels)
        self.colors = self._generate_colors()

        self.session = ort.InferenceSession(
    self.onnx_model,
    providers=["CPUExecutionProvider"],
)

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _validate_files(self):
        if not os.path.exists(self.onnx_model):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_model}")

        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(f"YAML file not found: {self.data_yaml}")

        model_size = os.path.getsize(self.onnx_model)

        if model_size < 1024 * 1024:
            raise ValueError(
                f"ONNX model file is too small: {model_size} bytes. "
                "This usually means best.onnx was not uploaded correctly "
                "or it is only a Git LFS pointer file."
            )

    def _load_labels(self):
        with open(self.data_yaml, mode="r", encoding="utf-8") as file:
            data = yaml.load(file, Loader=SafeLoader)

        if not data:
            raise ValueError("data.yaml is empty or invalid.")

        if "names" not in data:
            raise KeyError("data.yaml must contain a 'names' field.")

        labels = data["names"]

        if isinstance(labels, dict):
            labels = list(labels.values())

        if not isinstance(labels, list) or len(labels) == 0:
            raise ValueError("'names' in data.yaml must be a non-empty list.")

        return labels

    def predictions(self, image):
        if image is None:
            raise ValueError("Input image is None.")

        original_h, original_w = image.shape[:2]

        max_side = max(original_h, original_w)
        input_image = np.zeros((max_side, max_side, 3), dtype=np.uint8)
        input_image[0:original_h, 0:original_w] = image

        blob = cv2.dnn.blobFromImage(
            input_image,
            scalefactor=1 / 255.0,
            size=(self.input_size, self.input_size),
            mean=(0, 0, 0),
            swapRB=True,
            crop=False,
        )

        preds = self.session.run(
            self.output_names,
            {self.input_name: blob},
        )[0]

        detections = preds[0]

        boxes = []
        confidences = []
        class_ids = []

        x_factor = max_side / self.input_size
        y_factor = max_side / self.input_size

        for detection in detections:
            object_confidence = float(detection[4])

            if object_confidence < self.conf_threshold:
                continue

            class_scores = detection[5:]
            class_id = int(np.argmax(class_scores))
            class_score = float(class_scores[class_id])

            if class_score < self.score_threshold:
                continue

            confidence = object_confidence * class_score

            cx, cy, w, h = detection[0:4]

            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

        phone_count = 0

        if boxes:
            indices = cv2.dnn.NMSBoxes(
                boxes,
                confidences,
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
            )

            if len(indices) > 0:
                indices = np.array(indices).flatten()

                for index in indices:
                    x, y, w, h = boxes[index]
                    confidence = confidences[index]
                    class_id = class_ids[index]

                    if class_id >= len(self.labels):
                        continue

                    class_name = str(self.labels[class_id])
                    color = self.colors[class_id]

                    if class_name.lower() in ["phone", "cell phone", "cellphone", "mobile"]:
                        phone_count += 1

                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)

                    label = f"{class_name}: {int(confidence * 100)}%"

                    cv2.rectangle(
                        image,
                        (x, y),
                        (x + w, y + h),
                        color,
                        2,
                    )

                    text_top = max(0, y - 30)

                    cv2.rectangle(
                        image,
                        (x, text_top),
                        (x + max(w, 120), y),
                        color,
                        -1,
                    )

                    cv2.putText(
                        image,
                        label,
                        (x, max(15, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )

        return image, phone_count

    def _generate_colors(self):
        np.random.seed(10)
        colors = np.random.randint(100, 255, size=(self.nc, 3)).tolist()
        return [tuple(color) for color in colors]
