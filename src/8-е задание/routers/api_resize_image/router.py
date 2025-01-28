# python
import io
import json

# 3rdparty
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from PIL import Image
from pydantic import TypeAdapter

# project
from src.schemas.service_config import ServiceConfig
from src.schemas.service_output import ServiceOutput
from src.tools.logging_tools import get_logger

logger = get_logger()

service_config = r"src\configs\service_config.json"

with open(service_config, "r") as json_service_config:
    service_config_dict = json.load(json_service_config)

logger.info(f"Конфигурация сервиса: {service_config}")

service_config_adapter = TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(service_config_dict)

router = APIRouter(tags=["Main FastAPI service router"], prefix="")


@router.post(
    "/resize_image",
    summary="Выполняет ресайз изображения",
    response_description="Информация параметрах отресайзнутого изображения",
    description="Выполняет ресайз входного изображения",
    response_model=ServiceOutput,
)
async def resize_image(image: UploadFile = File(...)) -> ServiceOutput:
    """Метод для ресайза изображения

    Параметры:
        * `image` (`UploadFile`, optional): объект изображения

    Возвращает:
        * `ServiceOutput`: объект `ServiceOutput` c параметрами отресайзнутого изображения
    """
    image_content = await image.read()
    cv_image = np.array(Image.open(io.BytesIO(image_content)))

    logger.info(f"Принята картинка размерности: {cv_image.shape}")

    target_width = service_config_python.service_params.target_width
    target_height = service_config_python.service_params.target_height

    if len(cv_image.shape) == 2:
        cv_image = np.expand_dims(cv_image, axis=0)
        cv_image = np.transpose(cv_image, (1, 2, 0))

    resized_cv_image = cv2.resize(cv_image, (target_width, target_height))

    logger.info(
        f"Выполнен ресайз картинки до целевой размерности {target_height, target_width}"
    )

    service_output = ServiceOutput()
    service_output.width = resized_cv_image.shape[1]
    service_output.height = resized_cv_image.shape[0]
    service_output.channels = (
        resized_cv_image.shape[2] if len(resized_cv_image.shape) == 3 else 1
    )

    return service_output
