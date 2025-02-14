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


class ModelParams(BaseModel):
    """Датакласс, описывающий параметры детектора"""
    name: str = Field(default="yolo8_detector")
    """Название модели"""
    model_path: str = Field(
        default=r".\models\detectors\yolo8_detector.onnx")
    """Путь к модели"""


class ServiceConfig(BaseModel):
    """Конфигурация сервиса детектирования"""
    service_params: ServiceParams = Field(default=ServiceParams())
    """Основные параметры сервиса"""
    detector_Yolo8: ModelParams = Field(ModelParams)
    """Параметры детекторов"""
    classifier_ResNet18: ModelParams = Field(ModelParams)
    """Параметры классификаторов"""
    logging_params: LoggingParams = Field(default=LoggingParams())
    """Параметры логирования"""
    common_params: CommonParams = Field(default=CommonParams())
    """Общие настройки сервиса (хост, порт)"""
