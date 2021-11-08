

def caesar_cipher(text: str, k: int) -> str:
    ciphered = ""
    e = "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z".split(" ")
    for char in text:
        index = 0
        for elem in e:
            if elem == char.upper():
                ciphered += e[(index + k) % 26]
            elif char == " ":
                ciphered += " "
                break
            elif char.upper() not in e:
                ciphered += "%"
                break
            else:
                index += 1
    return ciphered


def main():
    print(caesar_cipher("szyfr juliusza cezara", 3))


if __name__ == "__main__":
    main()
