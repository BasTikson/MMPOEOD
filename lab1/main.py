"""
2. Лифт
В лифте есть 9 кнопок, соответствующих этажам, и кнопка открытия/закрытия дверей.
Нажатием на каждую кнопку сопровождается движение лифта на соответствующий этаж.
Движение от этажа к этажу осуществляется в течение одной минуты (в условных единицах времени).
Двери могут открываться во время остановки и не могут – во время движения.
При нажатии кнопки этажа с открытыми дверьми двери автоматически закрываются.
С одного этажа на один и тот же этаж лифт в движение не приводится.
Счётчик этажей реализовать визуально. Начальная конфигурация: лифт на первом этаже двери открыты.
"""

import time

class Elevator:
    def __init__(self):
        self.current_floor = 1
        self.doors_open = True
        self.moving = False

    def move_to_floor(self, target_floor):
        if self.current_floor == target_floor:
            print(f"Лифт уже на этаже {self.current_floor}. Движение не требуется.")
            return

        if self.doors_open:
            self.close_doors()

        self.moving = True
        print(f"Лифт начинает движение с этажа {self.current_floor} на этаж {target_floor}.")

        while self.current_floor != target_floor:
            if self.current_floor < target_floor:
                self.current_floor += 1
            else:
                self.current_floor -= 1
            print(f"Лифт на этаже {self.current_floor}.")
            time.sleep(1)  # Симуляция времени движения между этажами

        self.moving = False
        self.open_doors()

    def open_doors(self):
        if not self.moving:
            self.doors_open = True
            print("Двери лифта открыты.")
        else:
            print("Двери не могут быть открыты во время движения.")

    def close_doors(self):
        if self.doors_open:
            self.doors_open = False
            print("Двери лифта закрыты.")

    def press_floor_button(self, floor):
        if 1 <= floor <= 9:
            self.move_to_floor(floor)
        else:
            print("Неверный этаж. Выберите этаж от 1 до 9.")

    def press_door_button(self):
        if self.doors_open:
            self.close_doors()
        else:
            self.open_doors()

def main():
    elevator = Elevator()
    print("Лифт на первом этаже. Двери открыты.")

    while True:
        print("\nВыберите действие:")
        print("1-9: Вызвать лифт на соответствующий этаж")
        print("D: Открыть/закрыть двери")
        print("Q: Выйти")

        choice = input("Ваш выбор: ").strip().upper()

        if choice == 'Q':
            print("Выход из программы.")
            break
        elif choice == 'D':
            elevator.press_door_button()
        elif choice.isdigit() and 1 <= int(choice) <= 9:
            elevator.press_floor_button(int(choice))
        else:
            print("Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    main()