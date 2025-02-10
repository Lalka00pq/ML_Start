# python
import argparse
from typing import Any
# 3rdarty
import cv2
import torch
import numpy as np
# project
import os


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


def show_image_results(img: Any, classes_name: list, predicted: Any) -> None:
    '''Метод для отображения результата классификации изображения
    Args:
        classes_name (list): Список классов
        predicted (Any): Предсказание класса
    Returns: None : Выводит результат классификации
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2
    cv2.putText(img, classes_name[predicted.item()],
                org, font, fontScale, color, thickness)
    cv2.imshow('image', img)
    print(f"Predicted class: {classes_name[predicted.item()]}")


# исправить аннотацию типов
def inference_classifier(classifier: str, device: str, path_to_image: str) -> str:
    """Метод для инференса классификатора на единичном изображении

    Args:
        classifier (Any): Классификатор для получения предсказания
        path_to_image (str): Путь к изображению

    Returns:
        str: Название класса объекта на изображении
    """
    model = classifier
    model.eval()
    transform = OpenCVTransform()
    img = cv2.imread(path_to_image)
    img_resize = cv2.resize(img, (500, 500))
    image_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = transform(image_color).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted, img_resize


def load_classifier(
    name_of_classifier: str, path_to_pth_weights: str, device: str
) -> Any:
    """Метод для загрузки класификатора
    Args:
        name_of_classifier (str): Название классификатора
        path_to_pth_weights (str): Путь к PTH-файлу с весами классификатора
        device (str): Устройство для классификатора (cuda/cpu)
    Returns:
        Any: Классификатор и устройство
    """
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA доступна")
    else:
        device = torch.device("cpu")
        print("CPU доступен")

    available_names = ['resnet18',
                       'regnet',
                       'efficientnet',]
    name_of_classifier = name_of_classifier.lower()
    if name_of_classifier in available_names:
        print(f'Загружен классификатор {name_of_classifier}')
        classifier = torch.load(path_to_pth_weights)
        classifier.to(device)
        return classifier, device  # исправить данный момент
    else:  # переделать на исключение
        print(f"Классификатор {name_of_classifier} не найден")
        return None, None


def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов

    Returns:
        argparse.Namespace: _description_
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения классификатора на единичном изображении или папке с изображениями"
    )
    parser.add_argument(
        "--name_of_classifier", "-nc",
        type=str, help="Название классификатора",
        choices=['resnet18', 'efficientnet', 'regnet'], default='resnet18'
    )
    parser.add_argument(
        "--path_to_weights",
        "-wp",
        type=str,
        help="Путь к PTH-файлу с весами классификатора",
        default='.\\best_model_ResNet18_2.pth'
    )
    parser.add_argument(
        "--path_to_content",
        "-cp",
        type=str,
        help="Путь к одиночному изображению/папке с изображениями",
        # default='.\\boat40_ship_156_123.jpg'
        default='.\\test_dir'
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

    name_of_classifier = args.name_of_classifier
    path_to_weights = args.path_to_weights
    path_to_content = args.path_to_content
    use_cuda = args.use_cuda

    print(f"Название классификатора: {name_of_classifier}")
    print(f"Путь к файлам: {path_to_content}")
    print(f"Путь к классификатору модели: {path_to_weights}")
    classes_name = ['aircraft', 'ship']
    if use_cuda:
        print("Device: CUDA")
        device = 'cuda'
    else:
        print("Device: CPU")
        device = 'cpu'

    classifier, device_ = load_classifier(name_of_classifier=name_of_classifier,
                                          path_to_pth_weights=path_to_weights,
                                          device=device)

    if os.path.isfile(args.path_to_content):
        pred, img = inference_classifier(classifier=classifier,
                                         device=device_, path_to_image=path_to_content)
        show_image_results(img=img, classes_name=classes_name, predicted=pred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif os.path.isdir(args.path_to_content):
        images = [os.path.join(args.path_to_content, img) for img in os.listdir(
            args.path_to_content) if img.endswith(('.png', '.jpg', '.jpeg'))]

        if images:
            current_image_index = 0

            while True:
                image_path = images[current_image_index]
                pred, img = inference_classifier(classifier=classifier,
                                                 device=device_, path_to_image=image_path)
                show_image_results(
                    img=img, classes_name=classes_name, predicted=pred)
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
