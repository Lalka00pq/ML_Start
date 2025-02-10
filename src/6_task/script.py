# python
import argparse
from typing import Any
# 3rdarty
import cv2
import torch
import numpy as np
# project
import os

from ultralytics import YOLO


class OpenCVTransform:
    '''
    Класс трансформации изображения
    '''

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img)


def show_image_results(img: Any) -> None:  # изменить аннотацию типов
    '''Метод для отображения результата детекции изображения
    Args:
        classes_name (list): Список классов
        predicted (Any): Предсказание класса
    Returns: None : Выводит результат детектора на экран
    '''
    cv2.imshow('image', img)


def inference_detector(detector: str, device: str, path_to_image: str) -> Any:
    """Метод для интерфейса детектор на единичном изображении

    Args:
        classifier (Any): Детектор для получения предсказания
        path_to_image (str): Путь к изображению

    Returns:
        Any: Изображение
    """
    model = detector
    model.to(device)
    results = model(path_to_image)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            print(box.xyxy)
    img = results[0].plot()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def load_classifier(
    name_of_detector: str, path_to_pt_weights: str, device: str
) -> Any:  # изменить аннотацию
    """Метод для загрузки детектора
    Args:
        name_of_classifier (str): Название Детектора
        path_to_pth_weights (str): Путь к PTH-файлу с весами детектора
        device (str): Устройство для детектора (cuda/cpu)
    Returns:
        Any: Детектор и устройство
    """
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA доступна")
    else:
        device = torch.device("cpu")
        print("CPU доступен")

    available_names = ['yolo8',
                       'yolo10',
                       'yolo11',]
    name_of_detector = name_of_detector.lower()
    if name_of_detector in available_names:
        print(f'Загружен детектор {name_of_detector}')
        detector = YOLO(path_to_pt_weights)
        detector.to(device)
        return detector, device
    else:
        print(f"Детектор {name_of_detector} не найден")
        return None, None


def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов

    Returns: 
        argparse.Namespace: _description_ добавить описание
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения детектора на единичном изображении или папке с изображениями"
    )
    parser.add_argument(
        "--name_of_detector", "-nd",
        type=str, help="Название детектора",
        choices=['yolo8', 'yolo10', 'yolo11'], default='yolo8'
    )
    parser.add_argument(
        "--path_to_weights",
        "-wp",
        type=str,
        help="Путь к PT-файлу с детекторами",
        default=r'.\best_yolo8.pt'
    )
    parser.add_argument(
        "--path_to_content",
        "-cp",
        type=str,
        help="Путь к одиночному изображению/папке с изображениями",
        default=r'.\test_dir'        # default=r'.\images.jpg'
    )
    parser.add_argument(
        "--use_cuda",
        "-uc",
        action="store_true",
        help="Использовать ли CUDA для инференса",
    )
    args = parser.parse_args()

    return args


def main() -> None:
    """Основная логика работы с классификатором"""
    args = arguments_parser()

    name_of_detector = args.name_of_detector
    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content
    use_cuda = args.use_cuda

    print(f"Название классификатора: {name_of_detector}")
    print(f"Путь к файлам: {path_to_content}")
    print(f"Путь к классификатору модели: {path_to_weights}")

    if use_cuda:
        print("Device: CUDA")
        device = 'cuda'
    else:
        print("Device: CPU")
        device = 'cpu'

    detector, device_ = load_classifier(name_of_detector=name_of_detector,
                                        path_to_pt_weights=path_to_weights,
                                        device=device)

    if os.path.isfile(args.path_to_content):
        img = inference_detector(detector=detector, device=device_,
                                 path_to_image=path_to_content)
        show_image_results(img=img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif os.path.isdir(args.path_to_content):
        images = [os.path.join(args.path_to_content, img) for img in os.listdir(
            args.path_to_content) if img.endswith(('.png', '.jpg', '.jpeg'))]

        if images:
            current_image_index = 0

            while True:
                image_path = images[current_image_index]
                img = inference_detector(detector=detector,
                                         device=device_, path_to_image=image_path)
                show_image_results(
                    img=img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                elif key == ord('a') and current_image_index > 0:
                    current_image_index -= 1
                elif key == ord('d') and current_image_index < len(images) - 1:
                    current_image_index += 1
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
