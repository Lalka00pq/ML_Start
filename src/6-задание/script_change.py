import os


def replace_first_column(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    with open(input_file, 'w') as f:
        for line in lines:
            if line.startswith('1 '):
                f.write('0' + line[1:])
            else:
                f.write(line)


def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            input_file = os.path.join(directory, filename)
            replace_first_column(input_file)
            print(f"Processed {input_file}")


if __name__ == "__main__":
    directory = r'.\combine\train'
    process_directory(directory)
