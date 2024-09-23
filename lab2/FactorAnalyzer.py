import pandas as pd
import math
import os
import re



z_arr = ['z1', 'z2', 'z3', 'z4']
z_interactions_factors = ['z1z2', 'z1z3', 'z1z4', 'z2z3', 'z2z4', 'z3z4', 'z1z2z3', 'z1z2z4', 'z1z3z4', 'z2z3z4', 'z1z2z3z4'][::-1]


def get_elements_in_string(str_input):
    string_elements = [str_input[i:i+2] for i in range(0, len(str_input), 2)]
    string_set = set(string_elements)
    elements_set = set(z_arr)
    result_output = string_set.intersection(elements_set)
    return result_output

class FactorAnalyzer:
    def __init__(self, variant: int, file_input_2_3:str, file_input_2_4:str):

        self.list_b = {}

        # Информация по вариантам
        self.variant_2_4_data_scaled = pd.DataFrame()
        self.variant_2_4_data_witch_value = pd.DataFrame()
        self.variant_2_4_data = pd.DataFrame()
        self.variant_2_3_data = pd.DataFrame()

        # Общая информация
        self.data_2_4 = pd.DataFrame()
        self.data_2_3 = pd.DataFrame()

        # Пути до файлов
        self.file_input_2_3 = file_input_2_3
        self.file_input_2_4 = file_input_2_4

        self.variant = variant
        self.collect_variant_input_data()

    def read_input_xlsx(self):
        """
        Функция по варианту собирает входные данные для расчета Факторного анализа.
        :return:
        """

        try:
            self.data_2_3 = pd.read_excel(self.file_input_2_3, engine='openpyxl')
            self.data_2_4 = pd.read_excel(self.file_input_2_4, engine='openpyxl')
        except FileNotFoundError:
            if os.path.exists(self.file_input_2_3):
                print(f"Файл {self.file_input_2_4} не найден.")
            else:
                print(f"Файл {self.file_input_2_3} не найден.")
        except pd.errors.EmptyDataError:
            if os.path.getsize(self.file_input_2_3) == 0:
                print(f"Файл {self.file_input_2_3} пуст.")
            else:
                print(f"Файл {self.file_input_2_4} пуст.")
        except Exception as e:
            print(f"Произошла ошибка при загрузке данных: {e}")

    def collect_variant_input_data(self):
        """
        Функция получает входные данные по варианту
        :return:
        """
        # Неплохо бы название колонок тут же нормлаьное дать
        self.read_input_xlsx()
        if (self.data_2_3.empty and self.data_2_3.empty) is False:

            # Получаем dt варианта из 2.3.xlsx
            row = self.data_2_3.loc[self.data_2_3['№ Варианта'] == float(self.variant)]
            if row.empty is False:
                start_index = row.index[0]
                self.variant_2_3_data = self.data_2_3.iloc[start_index:start_index+4]
                self.variant_2_3_data.columns = ['№ Варианта', 'Фактор', 'В', 'Н']
                self.variant_2_3_data = self.variant_2_3_data.reset_index(drop=True)
            else:
                print("Ошибка, в 2.3.xlsx не найден вариант: ", self.variant)

            # Получаем dt варианта из 2.4.xlsx
            row = self.data_2_4.loc[self.data_2_4['№ Варианта'] == float(self.variant)]
            if row.empty is False:
                start_index = row.index[0]
                self.variant_2_4_data = self.data_2_4.iloc[start_index:start_index + 16]
                self.variant_2_4_data.columns = ['№ Варианта', '№ Эксперимента', 'z1', 'z2','z3','z4','y1','y2','y3',]
                self.variant_2_4_data = self.variant_2_4_data.reset_index(drop=True)
            else:
                print("Ошибка, в 2.4.xlsx не найден вариант: ", self.variant)
        else:
            print("Датафреймы входных данных пусты!")

    def preprocessing_variants(self):
        """
        Функция расставляет правильные значение диапазонов, Н-нижняя граница, В-верхняя соответственно.
        :return:
        """

        self.variant_2_4_data_witch_value = self.variant_2_4_data.copy()

        for index, row in self.variant_2_4_data_witch_value.iterrows():
            for z in z_arr:
                bound = row[z] # Какая граница, верхняя(В) или нижняя(Н)
                value = self.variant_2_3_data.loc[(self.variant_2_3_data['Фактор'] == z)][bound]
                if not value.empty:
                    value_in_bound = value.values[0]
                else:
                    value_in_bound = None
                    print(f"Значение для {bound} границы, фактора: {z} не определенно!" )
                    
                self.variant_2_4_data_witch_value.loc[index, z] = value_in_bound

    @staticmethod
    def calculate_x_scaled(z_value, z_a):
        """
        Функция, вычисляющая x(а) 2.1
        :return:
        """

        z_upper = z_value[0]
        z_lower = z_value[1]
        alf_a = (z_upper - z_lower) / 2 # интервал варьирования фактора.
        alf_z = (z_upper + z_lower) / 2 # центр интервала варьирования фактора
        alf_x = (z_a - alf_z) / alf_a  # масштабированные переменные

        return alf_x

    def planning_matrix_calculate(self,):
        """
        Функция, которая стоит матрицу планирования в нашем случае для 4-х факторов,
        4-х измерений в каждом эксперименте и функции в виде полинома 4-й степени в новых (масштабированных переменных).

        :return:
        """

        self.preprocessing_variants()

        print("\n")
        print("Матрица значений фактов:")
        print(self.variant_2_3_data.to_markdown(index=False))


        print("\n")
        print("Матрица планирования ДО масштабирования данных:")
        print(self.variant_2_4_data_witch_value.to_markdown(index=False))

        # Создаем шаблон для матрицы планирования С масштабированными данными
        self.variant_2_4_data_scaled = self.variant_2_4_data_witch_value.copy()


        for i in z_interactions_factors:
            self.variant_2_4_data_scaled.insert(6, i, [""]*16)


        for index, row in self.variant_2_4_data_witch_value.iterrows():

            # Переводим значение в формат [-1; 1]. Масштабируем.
            for z in z_arr:
                row_value_facts = self.variant_2_3_data.loc[(self.variant_2_3_data['Фактор'] == z)]
                z_value = row_value_facts.values[0][-2:] if not row_value_facts.empty else [None, None] # [z_upper, z_lower]
                z_a = row[z]
                x_a = self.calculate_x_scaled(z_value, z_a)
                self.variant_2_4_data_scaled.at[index, z] = x_a

            # Заполняем взаимодействие факторов ['z1', 'z2', 'z3', 'z4']
            for z in z_interactions_factors:
                composition = 1
                list_z_facts = get_elements_in_string(z)
                for i in list_z_facts:
                    multiplier = self.variant_2_4_data_scaled.loc[index, i]
                    composition *= multiplier
                self.variant_2_4_data_scaled.at[index, z] = composition

        print("\n")
        print("Матрица планирования ПОСЛЕ масштабирования данных:")
        print(self.variant_2_4_data_scaled.to_markdown(index=False))

    def calculate_y_arithmetic_mean(self):
        """
        Функция рассчитывает среднее арифметическое y.
        :return:
        """
        self.variant_2_4_data_scaled['y_среднее'] = self.variant_2_4_data_scaled[['y1', 'y2', 'y3']].mean(axis=1)
        print("\n")
        print("Матрица планирования после вычисления Y средне-арифметического:")
        print(self.variant_2_4_data_scaled.to_markdown(index=False))

    def calculate_b(self):
        self.planning_matrix_calculate()
        self.calculate_y_arithmetic_mean()
        list_y_arithmetic_mean = self.variant_2_4_data_scaled.loc[:, 'y_среднее'].to_list()
        list_z = z_arr + z_interactions_factors[::-1]
        k = len(z_arr)
        n = 2 ** k  # где k - это кол-во изучаемых факторов
        for index in range(len(list_y_arithmetic_mean)):
            y_mean = list_y_arithmetic_mean[index]
            for i in list_z:
                x_i = self.variant_2_4_data_scaled.loc[index, i]
                numbers = re.sub(r'\D', '', i)
                if f"b{numbers}" not in self.list_b.keys():
                    self.list_b[f"b{numbers}"] = 0
                self.list_b[f"b{numbers}"] += x_i*y_mean
        for index, value_b in self.list_b.items():
            self.list_b[index] = round(value_b / n, 3)

        print("\n")
        print("Коэффициенты уравнения регрессии:")
        for key, value in self.list_b.items():
            print(key, "=", value)

    def step_1st(self):
        """
        1. Рассчитать коэффициенты уравнения регрессии вида (2.5).
        В уравнении участвуют четыре типа коэффициентов при первой,
        второй, третей и четвертой степени.
        :return:
        """
        self.calculate_b()

    def step_2st(self):
        """
        2. Проверить полученные коэффициенты на значимость (выделить значимые и незначимые),
        с помощью критерия Стьюдента: если |b| > tкр * S, то соответствующий коэффициент b значим,
        иначе – нет и его полагают равным нулю.

        Уровень значимости для критической точки взять равным 0,05. tкр = t_cr = 0,05
        n – число экспериментов (число строк в матрице планирования)
        m – число опытов (наблюдений) в каждом эксперименте

        :return:
        """

        n = 3
        m = 16
        some_sum = 1



        print("\n")
        print("Проверка полученных коэффициентов на значимость:")
        for key, value in self.list_b.items():

            # Расчет tкр * S

            D_square = 1 / (n * (m - 1)) * some_sum
            S = math.sqrt(D_square / n * m)
