import numpy as np
import cv2
import argparse
from ultralytics import YOLO
import os


def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов

    Returns:
        argparse.Namespace: _description_
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения детектора на единичном изображении или папке с изображениями"
    )
    parser.add_argument(
        "--name_of_detector", "-nd",
        type=str,
        help="Имя детектора",
        default='yolo8'
    )
    parser.add_argument(
        "--path_to_model", "-pm",
        type=str,
        help="Путь к модели ONNX",
        default=r'.\best_yolo8 copy.onnx'
    )
    parser.add_argument(
        "--path_to_content", "-pc",
        type=str,
        help="Путь к изображению или папке с изображениями",
        default=r'.\images.jpg',
    )
    return parser.parse_args()


def inference_detector(detector, path_to_image: str) -> np.ndarray:
    """Метод для интерфейса детектор на единичном изображении

    Args:
        detector (ort.InferenceSession): Детектор для получения предсказания
        path_to_image (str): Путь к изображению

    Returns:
        np.ndarray: Изображение с предсказаниями
    """
    result = detector(path_to_image)
    for r in result:
        boxes = r.boxes
        for box in boxes:
            print(box.xyxy)
    img = result[0].plot()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def show_image(img: np.ndarray) -> None:
    """Метод для отображения изображения

    Args:
        img (np.ndarray): Изображение
    """
    cv2.imshow('Image', img)


def main() -> None:
    """Основная логика работы с классификатором"""
    args = arguments_parser()
    name_of_detector = args.name_of_detector
    path_to_weights = args.path_to_model
    path_to_content = args.path_to_content

    print(f"Название детектора: {name_of_detector}")
    print(f"Путь к файлам: {path_to_content}")
    print(f"Путь к детектору модели: {path_to_weights}")
    detector = YOLO(path_to_weights)
    if os.path.isfile(args.path_to_content):
        img = inference_detector(detector=detector,
                                 path_to_image=path_to_content)
        show_image(img)
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
                                         path_to_image=image_path)
                show_image(img)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break
                elif key == ord('a') and current_image_index > 0:
                    current_image_index -= 1
                elif key == ord('d') and current_image_index < len(images) - 1:
                    current_image_index += 1
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
