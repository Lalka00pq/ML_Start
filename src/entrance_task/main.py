def caesar_method(alphabet: list, shift: int, message: str) -> str:
    ''' Функция шифрования сообщения шифром Цезаря
    :param alphabet:
    :param shift:
    :param message:
    :return: new_message
    '''
    new_message = ""
    if not message:
        return 'String is empty'
    else:
        for i in message:
            try:
                new_index = (alphabet.index(i) + shift) % len(alphabet)
                new_message += alphabet[new_index]
            except ValueError:
                new_message = 'Error'
                break
        return new_message


def main():
    '''
    Главная функция
    :return:
    '''
    alphabet = list(' abcdefghijklmnopqrstuvwxyz')
    shift = int(input("Enter shift number (only integer): "))
    message = str(input("Enter message: ")).strip()
    result = caesar_method(alphabet,shift, message)
    print(result)


if __name__ == '__main__':
    main()