import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import os


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
        type=str,
        help="Имя классификатора",
        default='regnet'
    )
    parser.add_argument(
        "--path_to_model", "-pm",
        type=str,
        help="Путь к модели ONNX",
        default=r'.\model_resnet18_best_loss.onnx'
    )
    parser.add_argument(
        "--path_to_content", "-pc",
        type=str,
        help="Путь к изображению или папке с изображениями",
        default=r'.\boat40_ship_156_123.jpg',
    )
    return parser.parse_args()


def preprocess_image(image_path):
    input_image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch.numpy()


def inference_classifier(classifier, path_to_image):
    input_data = preprocess_image(path_to_image)
    input_name = classifier.get_inputs()[0].name
    output_name = classifier.get_outputs()[0].name
    outputs = classifier.run([output_name], {input_name: input_data})
    predicted_class = np.argmax(outputs[0], axis=1)
    return predicted_class[0], cv2.imread(path_to_image)


def show_image_results(img, classes_name, predicted):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (255, 255, 0)
    thickness = 2
    cv2.putText(img, classes_name[predicted.item()],
                org, font, fontScale, color, thickness)
    cv2.imshow("Result", img)
    cv2.waitKey(0)


def main():
    args = arguments_parser()
    path_to_model = args.path_to_model
    path_to_image = args.path_to_content
    classifier = ort.InferenceSession(path_to_model)
    classes_name = ["ship", "aircraft"]
    print(f"Используется классификатор {args.path_to_content}")
    if os.path.isfile(path_to_image):
        pred, img = inference_classifier(
            classifier=classifier, path_to_image=path_to_image)
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
                pred, img = inference_classifier(
                    classifier=classifier, path_to_image=image_path)
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
    else:
        print("В папке нет изображений")


if __name__ == '__main__':
    main()
