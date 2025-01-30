from fastapi import APIRouter, File, UploadFile, Path
from typing import Dict
from schemas.service_output import DetectedAndClassifiedObject, DetectedObject
from ultralytics import YOLO
from PIL import Image
import numpy as np
import onnxruntime as ort
from torchvision import transforms
import io
import os
import torch
import cv2
router = APIRouter(tags=["Find Objects"])


def preprocess_image(image_path: str) -> np.ndarray:
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch.numpy()


@router.post(
    '/find_objects',
    summary='Поиск объектов на изображении',
    description='Поиск объектов на изображении',
    response_description='HTTP Status Code 200',
    status_code=200,
    response_model=DetectedAndClassifiedObject)
def find_objects(
        image: UploadFile = File(...),
        path_to_detector: str = r'.\models\detectors\yolo8_detector.onnx',
        path_to_classifier: str = r'.\models\classifiers\regnet_classifier.onnx',
) -> DetectedAndClassifiedObject:
    """Метод поиска объектов на изображении
    Returns:
        Dict[str: str]: Словарь найденных объектов
    """
    classes_name = ['ship', 'aircraft']
    image = Image.open(io.BytesIO(image.file.read())).convert('RGB')
    orig_img = np.array(image)
    detector = YOLO(path_to_detector)
    detect_result = detector(image)
    classifier = ort.InferenceSession(path_to_classifier)
    # Начинается работа детектора
    detected_objects = []
    for box in detect_result[0].boxes:

        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

        confidence = box.conf.item()
        if confidence < 0.60:  # Порог доверия
            continue
        cropped_object = orig_img[int(ymin):int(ymax), int(xmin):int(xmax)]
        cropped_image = Image.fromarray(cropped_object)
        cropped_image.save('cropped_image.jpg')
    # ------------------------------------------------------------------------------
    # Начинается работа классификатора

        input_data = preprocess_image('cropped_image.jpg')
        ort_inputs = classifier.get_inputs()[0].name
        ort_outs = classifier.get_outputs()[0].name
        outputs = classifier.run([ort_outs], {ort_inputs: input_data})

        class_id = np.argmax(outputs[0])

        label = classes_name[class_id.item()]
        detected_objects.append(DetectedObject(
            label=label,
            confidence=confidence,
            xmin=int(xmin),
            ymin=int(ymin),
            xmax=int(xmax),
            ymax=int(ymax)
        ))

    return DetectedAndClassifiedObject(object_bbox=detected_objects)
