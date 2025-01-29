
from typing import Dict
from fastapi import APIRouter, status

router = APIRouter(tags=["Detector"])


@router.get(
    '/detectors',
    summary='Получение словаря детекторов',
    description='Получение словаря детекторов',
    response_description='HTTP Status Code 200',
    status_code=status.HTTP_200_OK,
    response_model=Getter)
def get_detectors() -> Dict[str: str]:
    """Метод получения словаря детекторов
    Returns:
        Dict[str: str]: Словарь детекторов
    """
    detectors: Dict[str: str] = {'detector_yolo8': 'best_yolo8 copy.onnx',
                                 'detector_yolo10n': 'best_yolo10n.onnx',
                                 'detector_yolo11 copy': 'best_yolo11 copy.onnx', }

    return detectors


# @router.get(
#     '/classifiers',
#     summary='Получение словаря классификаторов',

# )
