def caesar_method(alphabet: list, shift: int, message: str) -> str:
    ''' Функция шифрования сообщения шифром Цезаря
    :param alphabet (list): Алфавит для шифрования
    :param shift (int): Сдвиг для символов в алфавите
    :param message (str): Сообщение для шифрования(расшифрования)
    :return: new_message (str): Зашифрованное(Расшифрованное) сообщение
    '''
    new_message = ""
    for i in message:
        try:
            new_index = (alphabet.index(i) + shift) % len(alphabet)
            new_message += alphabet[new_index]
        except ValueError:
            new_message = 'Error'
            break
    return new_message


def main() -> None:
    '''
    Главная функция, вызывающая функцию шифрования
    :return: None
    '''
    alphabet = list(' abcdefghijklmnopqrstuvwxyz')
    shift = int(input("Enter shift number (only integer): "))
    message = str(input("Enter message: ")).strip()
    result = caesar_method(alphabet, shift, message)
    print(result)


if __name__ == '__main__':
    main()
