# python
from datetime import datetime

# 3rdparty
import pydantic


class HealthCheck(pydantic.BaseModel):
    """Датакласс для описания статуса работы нейросетевого сервиса"""

    status_code: int
    """Код статуса работы нейросетевого сервиса"""
    datetime: datetime
    """Отсечка даты и времени"""


class ServiceOutput(pydantic.BaseModel):
    """Датаконтракт выхода сервиса"""

    width: int = pydantic.Field(default=640)
    """Ширина преобразованного изображения"""
    height: int = pydantic.Field(default=480)
    """Высота преобразованного изображения"""
    channels: int = pydantic.Field(default=3)
    """Число каналов преобразованного изображения"""
