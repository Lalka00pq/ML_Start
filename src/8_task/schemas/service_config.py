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

    confidence: float = Field(default=0.75)
    """Коэффициент доверия"""
    available_detectors: dict = Field(default={
        'yolo8_detector': r'.\models\detectors\yolo8_detector.onnx',
        'yolo10n_detector': r'.\models\detectors\yolo10n_detector.onnx',
        'yolo11_detector': r'.\models\detectors\yolo11_detector.onnx',
    })
    '''Параметры детекторов'''
    available_classifiers: dict = Field(default={
        'efficientnet_classifier': r'.\models\classifiers\efficientnet_classifier.onnx',
        'regnet_classifier': r'.\models\classifiers\regnet_classifier.onnx',
        'resnet18_classifier': r'.\models\classifiers\resnet18_classifier.onnx',
    })
    """Параметры классификаторов"""


class ServiceConfig(BaseModel):
    """Конфигурация сервиса детектирования"""
    service_params: ServiceParams = Field(default=ServiceParams())
    """Основные параметры сервиса"""
    logging_params: LoggingParams = Field(default=LoggingParams())
    """Параметры логирования"""
    common_params: CommonParams = Field(default=CommonParams())
    """Общие настройки сервиса (хост, порт)"""
