from fastapi import APIRouter, File, UploadFile, Path
from typing import Dict
from schemas.service_output import DetectedAndClassifiedObject, DetectedObject
from ultralytics import YOLO
from PIL import Image
import numpy as np
import onnxruntime as ort
import io
import torch
router = APIRouter(tags=["Find Objects"])


@router.post(
    '/find_objects',
    summary='Поиск объектов на изображении',
    description='Поиск объектов на изображении',
    response_description='HTTP Status Code 200',
    status_code=200,
    response_model=DetectedAndClassifiedObject)
def find_objects(
        image: str = r'.\images.jpg',
        path_to_detector: str = r'.\models\detectors\yolo8_detector.onnx',
        path_to_classifier: str = r'.\models\classifiers\renset18_classifier.onnx',
) -> None:
    """Метод поиска объектов на изображении
    Returns:
        Dict[str: str]: Словарь найденных объектов
    """

    detector = YOLO(path_to_detector)
    detect_result = detector(image)

    detected_objects = []
    for box in detect_result[0].boxes:

        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

        confidence = box.conf.item()

        class_id = box.cls.item()

        label = detect_result[0].names[class_id]

        detected_objects.append(DetectedObject(
            label=label,
            confidence=confidence,
            xmin=int(xmin),
            ymin=int(ymin),
            xmax=int(xmax),
            ymax=int(ymax)
        ))
    return DetectedAndClassifiedObject(object_bbox=detected_objects)
