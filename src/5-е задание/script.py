# python
import argparse
from typing import Any
from torchvision import transforms
# 3rdarty
import cv2
import torch
# project
from PIL import Image
import numpy as np


class OpenCVTransform:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img):
        img = cv2.resize(img, (224, 224))  # Resize directly to 224x224
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # Convert to (C, H, W)
        return torch.from_numpy(img)


def show_image_results(classes_name: list, predicted: Any) -> None:
    '''Метод для отображения результата классификации изображения
    Args:
        classes_name (list): Список классов
        predicted (Any): Предсказание класса
    Returns: None : Выводит результат классификации
    '''
    print(f"Predicted class: {classes_name[predicted.item()]}")


def inference_classifier(classifier: str, device: str, path_to_image: str) -> str:
    """Метод для инференса классификатора на единичном изображении

    Args:
        classifier (Any): Классификатор для получения предсказания
        path_to_image (str): Путь к изображению

    Returns:
        str: Название класса объекта на изображении
    """
    model = torch.load(classifier, map_location=device)
    model.eval()
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
    #                          0.229, 0.224, 0.225])
    # ])
    # Переделать через OpenCV
    transform = OpenCVTransform()
    img = cv2.imread(path_to_image)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted


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
    # Можно написать короче
    if device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    available_names = ['resnet18',
                       'regnet',
                       'efficientnet',]
    name_of_classifier = name_of_classifier.lower()
    if name_of_classifier in available_names:
        print(f'Загружен классификатор {name_of_classifier}')
        classifier = torch.load(path_to_pth_weights)
        classifier.to(device)
    else:
        print(f"Классификатор {name_of_classifier} не найден")

    return classifier, device


def arguments_parser() -> argparse.Namespace:
    """Парсер аргументов

    Returns:
        argparse.Namespace: _description_
    """
    parser = argparse.ArgumentParser(
        description="Скрипт для выполнения классификатора на единичном изображении или папке с изображениями"
    )
    parser.add_argument(
        "--name_of_classifier", "-nc", type=str, help="Название классификатора"
    )
    parser.add_argument(
        "--path_to_weights",
        "-wp",
        type=str,
        help="Путь к PTH-файлу с весами классификатора",
    )
    parser.add_argument(
        "--path_to_content",
        "-cp",
        type=str,
        help="Путь к одиночному изображению/папке с изображениями",
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

    print(f"Name of classifier: {name_of_classifier}")
    print(f"Path to content: {path_to_content}")
    print(f"Path to weights: {path_to_weights}")

    if use_cuda:
        print("Device: CUDA")
    else:
        print("Device: CPU")
    classifier, device = load_classifier(name_of_classifier=name_of_classifier,
                                         path_to_pth_weights=path_to_weights,
                                         device=use_cuda)
    pred = inference_classifier(classifier=classifier,
                                device=device, path_to_image=path_to_content)


if __name__ == '__main__':

    pred = inference_classifier(classifier='./best_model_ResNet18_2.pth', device='cpu',
                                path_to_image='1.png')
    classes_name = ['aircraft', 'ship']
    show_image_results(classes_name=classes_name, predicted=pred)
