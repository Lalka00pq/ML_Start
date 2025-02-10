
import xml.etree.ElementTree as ET
from PIL import Image
import argparse

from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Парсер аргументов

    Returns:
        argparse.Namespace: Аргументы командной строки
    """
    parser = argparse.ArgumentParser(
        description="Парсер для файлов xml"
    )

    parser.add_argument(
        "--xml_path", '-xm',
        type=str,
        help="Путь к файлу аннотация xml",
        required=True
    )
    parser.add_argument(
        "--img_dir", '-imgd',
        type=str,
        help="Путь к директории с изображениями, для которых сделана аннотация",
        required=True
    )
    parser.add_argument(
        "--output_dir", '-outd',
        type=str,
        help="Директория в которую будут сохраняться изображения",
        required=True
    )

    return parser.parse_args()


def parse_annotations(xml_path: str, img_dir: str, output_dir: str) -> None:
    """
    Парсер для файлов xml.
    :param xml_path (str): Путь к файлу аннотации xml.
    :param img_dir (str): Путь к директории с изображениями, для которых сделана аннотация.
    :param output_dir (str): Директория в которую будут сохраняться изображения.
    return: None
    """
    xml_path = Path(xml_path)
    img_dir = Path(img_dir)
    output_dir = Path(output_dir)

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

                cropped_img = img.crop((xtl, ytl, xbr, ybr))

                if cropped_img.mode != "RGB":
                    cropped_img = cropped_img.convert("RGB")

                label_dir = output_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)

                output_file = label_dir / \
                    f"{Path(filename).stem}_{label}_{xtl}_{ytl}.jpg"
                cropped_img.save(output_file)

                print(f"Saved: {output_file}")


if __name__ == "__main__":
    args = parse_args()
    parse_annotations(args.xml_path, args.img_dir, args.output_dir)
