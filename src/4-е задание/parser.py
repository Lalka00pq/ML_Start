
import xml.etree.ElementTree as ET
from PIL import Image

from pathlib import Path


def parse_annotations(xml_path: str, img_dir: str, output_dir: str):
    """
    Парсер для файлов xml.
    :param xml_path: Путь к файлу аннотации xml.
    :param img_dir: Путь к директории с изображениями, для которых сделана аннотация.
    :param output_dir: Директория в которую будут сохраняться изображения.
    """
    xml_path = Path(xml_path)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)

    # Ensure output directories exist
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    for image in root.findall("image"):
        filename = image.get("name")
        img_path = img_dir / filename

        if not img_path.exists():
            print(f"Image {filename} not found, skipping.")
            continue

        with Image.open(img_path) as img:
            for box in image.findall("box"):
                label = box.get("label")
                xtl = int(float(box.get("xtl")))
                ytl = int(float(box.get("ytl")))
                xbr = int(float(box.get("xbr")))
                ybr = int(float(box.get("ybr")))

                # Вырезаем изображения из картинок
                cropped_img = img.crop((xtl, ytl, xbr, ybr))

                if cropped_img.mode != "RGB":
                    cropped_img = cropped_img.convert("RGB")

                # Сохраняем изображение в папку
                label_dir = output_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)

                output_file = label_dir / \
                    f"{Path(filename).stem}_{label}_{xtl}_{ytl}.jpg"
                cropped_img.save(output_file)

                print(f"Saved: {output_file}")


if __name__ == "__main__":
    parse_annotations(
        xml_path="ML_Start\\src\\4-е задание\\annotations_train.xml",
        img_dir="C:\\Users\\User\\Desktop\\ships_imgs",
        output_dir="C:\\Users\\User\\Desktop"
    )
