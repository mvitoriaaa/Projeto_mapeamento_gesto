from main import train_local_classifier


def main():
    labels = train_local_classifier()
    print("Modelo treinado com sucesso para os gestos:")
    print(", ".join(label.upper() for label in labels))


if __name__ == "__main__":
    main()
