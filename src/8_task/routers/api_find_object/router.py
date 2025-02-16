from fastapi import APIRouter, File, UploadFile
from schemas.service_output import DetectedAndClassifiedObject, DetectedObject
from ultralytics import YOLO
from PIL import Image
from pydantic import TypeAdapter
from schemas.service_config import ServiceConfig
import numpy as np
import onnxruntime as ort
from torchvision import transforms
import io
import json
import os

from tools.logging_tools import get_logger
logger = get_logger()
service_config = r".\configs\service_config.json"
with open(service_config, "r") as json_service_config:
    service_config_dict = json.load(json_service_config)

logger.info(f"Конфигурация сервиса: {service_config}")

service_config_adapter = TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(
    service_config_dict)

router = APIRouter(tags=["Find Objects"], prefix="")


def preprocess_image(image_path: str) -> np.ndarray:
    """Предобработка изображения для классификатора

    Args:
        image_path (str): Путь к изображению

    Returns:
        np.ndarray: массив изображения
    """
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
        detector_name: str = service_config_python.detector_Yolo8.name,
        classifier_name: str = service_config_python.classifier_ResNet18.name,
        input_confidence: float = service_config_python.service_params.confidence
) -> DetectedAndClassifiedObject:
    """Метод поиска объектов на изображении
    Returns:
        Dict[str: str]: Словарь найденных объектов
    """
    detector_name = detector_name.lower()
    classifier_name = classifier_name.lower()
    path_to_detector = service_config_python.detector_Yolo8.model_path
    path_to_classifier = service_config_python.classifier_ResNet18.model_path
    classes_name = ['aircraft', 'ship']
    image = Image.open(io.BytesIO(image.file.read())).convert('RGB')
    orig_img = np.array(image)
    detector = YOLO(path_to_detector)
    detect_result = detector(image)
    classifier = ort.InferenceSession(path_to_classifier)
    logger.info(
        f"Загружены детектор - {detector.model} и классификатор: {path_to_classifier}"
    )
    # ------------------------------------------------------------------------------
    # Начинается работа детектора
    detected_objects = []
    for box in detect_result[0].boxes:

        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()

        confidence = box.conf.item()
        if confidence < input_confidence:  # Порог доверия
            continue
        cropped_object = orig_img[int(ymin):int(ymax), int(xmin):int(xmax)]
        cropped_image = Image.fromarray(cropped_object)
        cropped_image.save('cropped_image.jpg')
        logger.info(
            f"Обнаружен объект с координатами {xmin, ymin, xmax, ymax}")
    # ------------------------------------------------------------------------------
    # Начинается работа классификатора

        input_data = preprocess_image('cropped_image.jpg')
        if os.path.exists('cropped_image.jpg'):
            os.remove('cropped_image.jpg')
        ort_inputs = classifier.get_inputs()[0].name
        ort_outs = classifier.get_outputs()[0].name
        outputs = classifier.run([ort_outs], {ort_inputs: input_data})

        class_id = np.argmax(outputs[0])

        label = classes_name[class_id.item()]
        logger.info(
            f"Объект классифицирован как {label} с вероятностью {round(confidence*100)}%")
        detected_objects.append(DetectedObject(
            label=label,
            confidence=confidence,
            xmin=int(xmin),
            ymin=int(ymin),
            xmax=int(xmax),
            ymax=int(ymax)
        ))

    return DetectedAndClassifiedObject(object_bbox=detected_objects)
