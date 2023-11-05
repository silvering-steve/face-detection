import os
import wget
import streamlit as st

import cv2
import numpy as np
from ultralytics import YOLO


class Predict:
    MODEL_PATH = "models/head.pt"

    def __init__(self):
        if not os.path.isfile(self.MODEL_PATH):
            wget.download(st.secrets["MODEL_URL"], out="models")

        self.model = YOLO(self.MODEL_PATH)

    def __call__(self, image, *args, **kwargs):
        image_bytes = np.asarray(bytearray(image), dtype=np.uint8)
        image_array = cv2.imdecode(image_bytes, 1)

        result = self.model.predict(image_array, conf=kwargs['conf'], iou=kwargs['iou'])

        return result[0].plot()


predict = Predict()