import os


def replace_first_column(input_file: str) -> None:
    """Заменяет первый столбец в файле на 0, если он равен 1

    Args:
        input_file (str): Файл для обработки
    RETURN:
        None
    """
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(input_file, 'w') as f:
        for line in lines:
            if line.startswith('1 '):
                f.write('0' + line[1:])
            else:
                f.write(line)


def process_directory(directory: str) -> None:
    """Обрабатывает все файлы в директории

    Args:
        directory (str): Директория с файлами

    Return: None
    """
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            input_file = os.path.join(directory, filename)
            replace_first_column(input_file)
            print(f"Processed {input_file}")


if __name__ == "__main__":
    directory = r'.\combine\test'
    process_directory(directory)
