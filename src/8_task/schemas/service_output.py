# python
from datetime import datetime

# 3rdparty
import pydantic
from typing import List


class HealthCheck(pydantic.BaseModel):
    """Датакласс для описания статуса работы нейросетевого сервиса"""

    status_code: int
    """Код статуса работы нейросетевого сервиса"""
    datetime: datetime
    """Отсечка даты и времени"""


class DetectedObject(pydantic.BaseModel):
    '''Датакласс для описания найденного объекта'''
    label: str
    '''Название объекта'''
    confidence: float
    '''Вероятность детекции'''
    xmin: int
    '''Координата x левого верхнего угла'''
    ymin: int
    '''Координата y левого верхнего угла'''
    xmax: int
    '''Координата x правого нижнего угла'''
    ymax: int
    '''Координата y правого нижнего угла'''


class DetectedAndClassifiedObject(pydantic.BaseModel):
    """ Датакласс данных которые будут возвращены сервисом (детекция и классификация) """
    object_bbox: List[DetectedObject] | None
    """ Координаты объекта """
