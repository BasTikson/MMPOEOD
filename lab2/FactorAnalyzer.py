import pandas as pd
import math
import os
import re

z_arr = ['z1', 'z2', 'z3', 'z4']
z_interactions_factors = ['z1z2', 'z1z3', 'z1z4', 'z2z3', 'z2z4', 'z3z4', 'z1z2z3', 'z1z2z4', 'z1z3z4', 'z2z3z4', 'z1z2z3z4'][::-1]
n = 16
m = 3

def get_elements_in_string(str_input):
    string_elements = [str_input[i:i+2] for i in range(0, len(str_input), 2)]
    string_set = set(string_elements)
    elements_set = set(z_arr)
    result_output = string_set.intersection(elements_set)
    return result_output

def transform_key(key):
    key = key.replace('b', '')
    result = ''
    for char in key:
        if char.isdigit():
            result += 'z' + char
        else:
            result += char

    return result

class FactorAnalyzer:
    def __init__(self, variant: int, file_input_2_3:str, file_input_2_4:str):


        self.D_square = None # Дисперсия воспроизводимости
        self.S = None # Среднее квадратическое отклонение
        self.t_cr = 2.04 # тоже штука с инет атаблиц
        self.F_table = 2.9 # зять инет это для 13 дов
        self.count_significant_element = 0
        self.list_b = {}
        self.minor_parameters_b = []
        self.y_regression = [0] * n

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
        for index in range(len(list_y_arithmetic_mean)):
            y_mean = list_y_arithmetic_mean[index]
            if "b0" not in self.list_b.keys():
                self.list_b["b0"] = 0
            self.list_b["b0"] += y_mean
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

    def calculate_S(self):
        sum_mean_y = 0

        # Вычисление tкр * S
        for i in range(n):
            for j in range(m):
                y_mean = self.variant_2_4_data_scaled.loc[i, "y_среднее"]
                y_j = self.variant_2_4_data_scaled.loc[i, f"y{j + 1}"]
                sum_mean_y += (y_j - y_mean) ** 2
        print("\n")
        print("sum_mean_y: ", sum_mean_y)
        print("\n")
        self.D_square = (1 / (n * (m - 1))) * sum_mean_y
        print("\n")
        print("self.D_square: ", self.D_square)
        print("\n")

        self.S = math.sqrt(self.D_square /( n * m))
        print("\n")
        print(f"Значение критической точки S * t_cr = {round(self.S, 3)} * {self.t_cr} = {round(self.S * self.t_cr, 3)}")

    def get_values_regression(self, ):
        """
        Берет значение изучаемого параметра, вычисленное по
        уравнению регрессии со значимыми коэффициентами для i-го эксперимента;

        :return:
        """


        for i in range(n):
            for key in self.list_b.keys():
                if key not in self.minor_parameters_b:
                    if key == 'b0':
                        self.y_regression[i] = self.y_regression[i] + self.list_b['b0']
                    else:
                        key_for_data_scaled  = transform_key(key)
                        self.y_regression[i] = self.y_regression[i] + self.list_b[key] * self.variant_2_4_data_scaled.loc[i, key_for_data_scaled]

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
        self.calculate_S()


        print("\n")
        print("Проверка полученных коэффициентов на значимость:")
        for key, value in self.list_b.items():
            value = value * -1 if value < 0 else value
            if value < self.S * self.t_cr:
                print(f"Коэффициент {key} НЕ значимый!")
                self.minor_parameters_b.append(key)

            else:
                self.count_significant_element+=1
                print(f"Коэффициент {key} значимый!")

        print("\n")
        print(f"Количество значимых элементов = {self.count_significant_element}")

    def step_3st(self):
        """
        3. Проверить полученное уравнение регрессии со значимыми
        коэффициентами на адекватность. Уровень значимости взять также
        равным 0,05.

        Проверка на адекватность полученного уравнения регрессии со
        значимыми коэффициентами осуществляется с помощью критерия
        Фишера: если F_расч < F_табл, то уравнение адекватно,
        в противном случае – неадекватно.

        F_расч = F_calculated;
        F_табл = F_table = 0.05?;

        D_ост = D_residual(в данном случае в квадрате);
        n – число экспериментов;
        m – число опытов в каждом эксперименте;
        r – количество значимых коэффициентов в уравнении регрессии;
        y_i – значение изучаемого параметра, вычисленное по уравнению
                регрессии со значимыми коэффициентами для i-го эксперимента;


        :return:
        """
        self.get_values_regression()
        sum_mean_y_slash = 0

        if len(self.minor_parameters_b) != 0:
            print("\n")
            print(f"Проводим проверку на адекватность уравнения: ")
            for i in range(n):
                y_mean = self.variant_2_4_data_scaled.loc[i, "y_среднее"]
                sum_mean_y_slash += (self.y_regression[i] - y_mean) ** 2

            r = self.count_significant_element
            D_residual = m / (n - r) * sum_mean_y_slash
            F_calculated = D_residual / self.D_square

            if F_calculated < self.F_table:
                print(f"Уравнение адекватно")
            else:
                print(f"Уравнение НЕ адекватно")
        else:
            print("\n")
            print(f"Проверка на адекватность не требуется, так как коэффициентов незначащих элементов НЕТ")

    def step_4st(self):
        """
        4. Ранжировать факторы и их взаимодействия по степени влияния.
        :return:
        """

        list_b_without_minor_parameters = self.list_b.copy()
        for key in self.minor_parameters_b:
            list_b_without_minor_parameters.pop(key, None)

        print("\n")
        print(f"Ранжируем факторы по степени влияния значащих коэффициентов: ")

        sorted_list_tuple = sorted(list_b_without_minor_parameters.items(), key=lambda item: item[1], reverse=True)
        sorted_dict = dict(sorted_list_tuple)
        for key, value in sorted_dict.items():
            print(key, "=", value)

    def step_5st(self):
        """
        5. Получить уравнение в исходных переменных:
        y = f(z1, z2, z3, z4)
        :return:
        """

        equation = f"{self.list_b['b0']}"

        for key in self.list_b.keys():
            if key != 'b0' and self.list_b[key] != 0:
                indices = [int(index) for index in key if index.isdigit()]
                equation += f" + {self.list_b[key]}"

                for index in indices:
                    equation += f" * z{index}"

        print("\n")
        print("Уравнение в натуральных переменных:")
        print("y =", equation)

        z = {'1': [], '2': [], '3': [], '4': []}

        for i in range(n):
            for a in z_arr:
                x_value = self.variant_2_4_data_scaled.loc[i, a]
                numbers = re.sub(r'\D', '', a)
                b_value = self.list_b[f"b{numbers}"]
                row_value_facts = self.variant_2_3_data.loc[(self.variant_2_3_data['Фактор'] == a)]
                z_value = row_value_facts.values[0][-2:] if not row_value_facts.empty else [None, None]  # [z_upper, z_lower]

                z_upper = z_value[0]
                z_lower = z_value[1]

                if z_upper and z_lower:
                    alf_a = (z_upper - z_lower) / 2  # интервал варьирования фактора.
                    alf_z = (z_upper + z_lower) / 2  # центр интервала варьирования фактора
                    value_z = round(x_value * b_value * (alf_a + alf_z), 3)
                    z[str(numbers)].append(value_z)
                else:
                    z[str(numbers)].append(None)

        print("\n")
        print("Значения факторов:")

        for i in z.keys():
            print(f"z{i}", "=", z[i])


