from schemas.service_config import ClassifierParams, DetectorParams
from typing import Dict
from fastapi import APIRouter, status

router_detect = APIRouter(tags=["Detector"])
router_classifier = APIRouter(tags=["Classifier"])


@router_detect.get(
    '/detectors',
    summary='Получение словаря детекторов',
    description='Получение словаря детекторов',
    response_description='HTTP Status Code 200',
    status_code=status.HTTP_200_OK,
    response_model=DetectorParams)
def get_detectors() -> Dict[str, str]:
    """Метод получения словаря детекторов
    Returns:
        Dict[str: str]: Словарь детекторов
    """
    detectors: Dict[str, str] = {'detector_yolo8': 'best_yolo8 copy.onnx',
                                 'detector_yolo10n': 'best_yolo10n.onnx',
                                 'detector_yolo11 copy': 'best_yolo11 copy.onnx', }

    return detectors


@router_classifier.get(
    '/classifiers',
    summary='Получение словаря классификаторов',
    description='Получение словаря классификаторов',
    response_description='HTTP Status Code 200',
    status_code=status.HTTP_200_OK,
    response_model=ClassifierParams)
def get_classifiers() -> Dict[str, str]:
    """Метод получения словаря классификаторов
    Returns:
        Dict[str: str]: Словарь классификаторов
    """
    classifiers: Dict[str, str] = {'classifier_resnet18': 'model_resnet18_best_loss.onnx',
                                   'classifier_resnet34': 'model_regnet_best_loss.onnx',
                                   'classifier_resnet50': 'model_efficientnet_best_loss.onnx', }

    return classifiers
