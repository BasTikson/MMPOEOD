from main import ModelIdentification
def main():
    print("Восстановление функциональной зависимости по экспериментальным данным.")

    while True:
        choice = input("\nВведите вариант (от 1 до 40): ")
        try:
            variant = int(choice)
            if not (1 <= variant <= 40):
                raise ValueError("Вариант должен быть от 1 до 40.")
            break
        except ValueError as e:
            print("\n")
            print("Неправильный формат ввода данных. Попробуйте еще раз!")
            print(e)
    modelIn = ModelIdentification(variant)
    modelIn.run()

if __name__ == "__main__":
    main()