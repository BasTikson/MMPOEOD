from FactorAnalyzer import FactorAnalyzer

def main():
    print("Факторный анализ экспериментальных данных.")
    choice = input("\nВведите вариант: ")
    file_input_2_3 = "const/2.3.xlsx"
    file_input_2_4 = "const/2.4.xlsx"

    try:
        variant = int(choice)
        if 1 <= variant <= 40:
            fac_anal = FactorAnalyzer(variant, file_input_2_3, file_input_2_4)
            fac_anal.step_1st()
        else:
            print("\n")
            print("Выбран несуществующий вариант, выберите от 1 до 40!")
            main()
    except ValueError as e:
        print("\n")
        print("Неправильный формат ввода данных. Попробуйте еще раз!")
        print(e)
        main()
    except Exception as e:
        print("\n")
        print("Произошла непредвиденная ошибка!",)
        print(e)


if __name__ == "__main__":
    main()