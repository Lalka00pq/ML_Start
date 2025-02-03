from pydantic import BaseModel, Field


class LoggingParams(BaseModel):
    """Датакласс, описывающий настройки логирования"""

    save_logs: bool = Field(default=False)
    """Сохранять ли логи работы сервиса"""
    logs_directory: str = Field(default="")
    """Директория, в которую предполагается сохранять логи работы сервиса"""
    logging_config: str = Field(default="")
    """Путь к YAML-конфигурации логирования"""


class CommonParams(BaseModel):
    """Датакласс, описывающий общие настройки сервиса"""

    host: str = Field(default="localhost")
    """Адрес хоста сервиса"""
    port: int = Field(default=8000)
    """Порт сервиса"""


class ServiceParams(BaseModel):
    """Датакласс, описываемый основные параметры сервиса"""

    target_width: int = Field(default=1920)
    """Целевая ширина изображения"""
    target_height: int = Field(default=1080)
    """Целевая высота изображения"""


class ServiceConfig(BaseModel):
    """Конфигурация сервиса"""

    service_params: ServiceParams = Field(default=ServiceParams())
    """Основные параметры сервиса"""
    logging_params: LoggingParams = Field(default=LoggingParams())
    """Параметры логирования"""
    common_params: CommonParams = Field(default=CommonParams())
    """Общие настройки сервиса (хост, порт)"""


class DetectorParams(BaseModel):
    """Датакласс, описывающий параметры детектора"""

    detector_name: str = Field(default="Yolo11")
    """Название детектора"""
    detector_path: str = Field(
        default=r"\models\detectors\yolo11_detector.onnx")
    """Путь к файлу детектора (формат ONNX)"""


class ClassifierParams(BaseModel):
    """Датакласс, описывающий параметры классификатора"""

    classifier_name: str = Field(default="resnet18")
    """Название классификатора"""
    classifier_path: str = Field(
        default=r"\models\classifiers\resnet18_classifier.onnx")
    """Путь к файлу классификатора (формат ONNX)"""


class ServiceConfigDetection(BaseModel):
    """Конфигурация сервиса детектирования"""
    detector_params: DetectorParams = Field(default=DetectorParams())
    """Параметры детектора"""
    classifier_params: ClassifierParams = Field(default=ClassifierParams())
    """Параметры классификатора"""
    logging_params: LoggingParams = Field(default=LoggingParams())
    """Параметры логирования"""
    common_params: CommonParams = Field(default=CommonParams())
    """Общие настройки сервиса (хост, порт)"""
