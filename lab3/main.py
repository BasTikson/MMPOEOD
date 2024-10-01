import numpy as np
import matplotlib
import os
import pandas as pd
from scipy.stats import linregress
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

def read_excel_to_dataframe(file_path):
    """
    Читает файл Excel по указанному пути и возвращает датафрейм.

    :param file_path: Путь до файла Excel с расширением .xlsx
    :return: Датафрейм с данными из файла Excel или None в случае ошибки
    """
    if not os.path.exists(file_path):
        print(f"Файл не найден: {file_path}")
        return None

    if os.path.getsize(file_path) == 0:
        print(f"Файл пустой: {file_path}")
        return None

    try:
        df = pd.read_excel(file_path)
        if df.empty:
            print(f"Файл содержит пустой датафрейм: {file_path}")
            return None
        return df
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return None

def clean_data(df, i):
    """
    Очистка данных от NaN и 'nan' значений.

    :param df: DataFrame с данными.
    :param i: Индекс выборки.
    :return: Очищенные данные x_cleaned и y_cleaned.
    """
    x_col = f"X_{i}"
    y_col = f"Y_{i}"
    x = df[x_col].values
    y = df[y_col].values
    x[x == 'nan'] = np.nan
    y[y == 'nan'] = np.nan
    mask = ~np.isnan(x.astype(float)) & ~np.isnan(y.astype(float))
    x_cleaned = x[mask].astype(float)
    y_cleaned = y[mask].astype(float)
    return x_cleaned, y_cleaned

def hyperbolic_regression(x, y):
    def func(x, a, b):
        return a / x + b
    x[x == 0] = np.min(x[x != 0]) / 1000
    B, _ = curve_fit(func, x, y)
    return B

def logarithmic_regression(x, y, base=10):
    # Фильтруем данные, чтобы исключить нулевые и отрицательные значения
    mask = (x > 0) & (y > 0)
    x_filtered = x[mask]
    y_filtered = y[mask]

    # Проверка на наличие данных после фильтрации
    # if len(x_filtered) == 0 or len(y_filtered) == 0:
    #     raise ValueError("После фильтрации не осталось данных для выполнения регрессии.")

    # Замена переменных
    S = np.log(x_filtered) / np.log(base)
    R = np.log(y_filtered) / np.log(base)

    # Линейная регрессия для R и S
    slope, intercept, r_value, p_value, std_err = linregress(S, R)

    # Возвращаем коэффициенты
    b0 = intercept
    b1 = slope

    return b0, b1

def parabolic_regression(x, y):
    # Вычисляем математические ожидания
    m_X = np.mean(x)
    m_Y = np.mean(y)
    m_X2 = np.mean(x ** 2)
    m_X3 = np.mean(x ** 3)
    m_X4 = np.mean(x ** 4)
    m_XY = np.mean(x * y)
    m_X2Y = np.mean(x ** 2 * y)
    A = np.array([
        [1, m_X, m_X2],
        [m_X, m_X2, m_X3],
        [m_X2, m_X3, m_X4]
    ])
    C = np.array([m_Y, m_XY, m_X2Y])
    b0, b1, b2 = np.linalg.solve(A, C)
    return b0, b1, b2

def linear_regression(x, y):
    b0, b1, r_value, p_value, std_err = linregress(x, y)
    return b0,b1

def calculate_r_squared(y_true, y_pred):
    """
    Вычисляет коэффициент детерминации (R²).

    :param y_true: Истинные значения зависимой переменной.
    :param y_pred: Предсказанные значения зависимой переменной.
    :return: Коэффициент детерминации (R²).
    """
    # Вычисляем общую сумму квадратов (TSS)
    y_mean = np.mean(y_true)
    tss = np.sum((y_true - y_mean) ** 2)

    # Вычисляем сумму квадратов ошибок (RSS)
    rss = np.sum((y_true - y_pred) ** 2)

    # Проверка на корректность значений
    if tss == 0:
        raise ValueError("Общая сумма квадратов равна нулю, невозможно вычислить коэффициент детерминации.")

    # Вычисляем коэффициент детерминации (R²)
    r_squared = 1 - (rss / tss)

    return r_squared

def find_best_log_base(x, y):
    """
    Поиск лучшего основания логарифма для логарифмической модели.

    :param x: Значения X.
    :param y: Значения Y.
    :return: Лучшее основание логарифма и соответствующий коэффициент детерминации.
    """
    best_base = 10
    best_r_squared = -np.inf
    for base in range(2, 11):
        b0, b1 = logarithmic_regression(x, y, base=base)
        mask = (x > 0) & (y > 0)
        x_filtered = x[mask]
        y_filtered = y[mask]
        y_pred = b0 + b1 * np.log(x_filtered ) / np.log(base)
        r_squared = calculate_r_squared(y_filtered, y_pred)
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_base = base
    return best_base, best_r_squared

def evaluate_linear_model(x, y):
    """
    Оценка линейной модели: вычисление коэффициента корреляции и оценка силы и направления линейной зависимости.

    :param x: Значения X.
    :param y: Значения Y.
    :return: Коэффициент корреляции.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    correlation_coefficient = r_value
    print(f"Коэффициент корреляции для линейной модели: {correlation_coefficient}")
    if abs(correlation_coefficient) < 0.5:
        print("Линейная зависимость отсутствует")
    elif abs(correlation_coefficient) < 0.75:
        print("Линейная зависимость слабая")
    else:
        print("Линейная зависимость сильная")
    return correlation_coefficient

class ModelIdentification:
    def __init__(self, variant: int):

        self.hyperbolic_y_pred = {}
        self.parabolic_y_pred = {}
        self.linear_y_pred = {}

        self.coef_func_dependence = {}
        self.data = pd.DataFrame()
        self.variant = variant
        self.models = {}
        self.predictions = {}
        self.uploading_data()

    def uploading_data(self):
        """
        Функция выгружает информацию по варианту из таблиц 3.5, 3.6

        :return:
        """
        path = f"const/3.5.xlsx"
        df = pd.read_excel(path)
        row = df.loc[df['№ варианта'] == int(self.variant)]

        if not row.empty:
            start_index = row.index[0]
            df = df.iloc[start_index:start_index + 30]
            df_trimmed = df.drop('№ варианта', axis=1)
            df_trimmed = df_trimmed.reset_index(drop=True)
            df_trimmed.columns = ["№ Измерения", "X_1", "Y_1", "X_2", "Y_2", "X_3", "Y_3"]
            self.data = df_trimmed

        path = f"const/3.6.xlsx"
        df = pd.read_excel(path)
        row = df.loc[df['№ варианта'] == int(self.variant)]
        if not row.empty:
            start_index = row.index[0]
            df = df.iloc[start_index:start_index + 17]
            df_trimmed = df.drop('№ варианта', axis=1)
            df_trimmed.columns = ["i_1", "x_1", "y_1", "i_2", "x_2", "y_2", "i_3", "x_3", "y_3"]
            arr_1 = df_trimmed[["x_1", "y_1"]]
            arr_2 = df_trimmed[["x_2", "y_2"]]
            arr_3 = df_trimmed[["x_3", "y_3"]]
            arr_2.columns = ['x_1', 'y_1']
            arr_3.columns = ['x_1', 'y_1']
            df_result = pd.concat([arr_1, arr_2, arr_3], axis=0)
            df_result = df_result.reset_index(drop=True)
            df_result.columns = ["X_4", "Y_4"]
            self.data = pd.concat([self.data, df_result], axis=1)
        self.data = self.data.drop("№ Измерения", axis=1)
        print("\n")
        print("Выборки")
        print(self.data)

    def plot_data(self):
        """
        Постройка графиков для каждой выборки.
        :return:
        """
        df = self.data
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Список пар (X, Y) и соответствующих им цветов
        data_pairs = [
            ('X_1', 'Y_1', 'b'),
            ('X_2', 'Y_2', 'g'),
            ('X_3', 'Y_3', 'r'),
            ('X_4', 'Y_4', 'c')
        ]

        # Цикл для создания графиков
        for i, (x_col, y_col, color) in enumerate(data_pairs):
            row = i // 2
            col = i % 2
            axs[row, col].plot(df[x_col], df[y_col], marker='o', linestyle='-', color=color,
                               label=f'{x_col} vs {y_col}')
            axs[row, col].set_title(str(i + 1))
            axs[row, col].set_xlabel(x_col)
            axs[row, col].set_ylabel(y_col)
            axs[row, col].legend()
            axs[row, col].grid(True)

        plt.tight_layout()

        # Отображение графика
        plt.show()

    def select_best_model(self, x, y, models):
        """
        Выбор лучшей модели по коэффициенту детерминации.

        :param x: Значения X.
        :param y: Значения Y.
        :param models: Словарь моделей.
        :return: Лучшая модель, её параметры и коэффициент детерминации.
        """
        best_model = None
        best_r_squared = -np.inf
        best_params = None

        for model_name, model_func in models.items():

            try:
                if model_name == "Logarithmic":
                    best_base, best_r_squared_log = find_best_log_base(x, y)
                    params = model_func(x, y, base=best_base)
                    r_squared = best_r_squared_log

                elif model_name == "Hyperbolic":
                    params = model_func(x, y)
                    b0, b1 = params
                    y_pred = b0 / x + b1
                    r_squared = calculate_r_squared(y, y_pred)

                elif model_name == "Parabolic":
                    params = model_func(x, y)
                    b0, b1, b2 = params
                    y_pred = b0 + b1 * x + b2 * x ** 2
                    r_squared = calculate_r_squared(y, y_pred)

                else:
                    params = model_func(x, y)
                    b0, b1 = params
                    y_pred = b0 + b1 * x
                    r_squared = calculate_r_squared(y, y_pred)

                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    best_model = model_name
                    best_params = params

            except Exception as e:
                print(f"Ошибка в модели {model_name}: {e}")

        return best_model, best_params, best_r_squared

    def step_1st(self):
        self.plot_data()

    def step_2st(self):
        """
        2. Параметрическая идентификация модели. Для каждой выборки из таблицы 3.5 вычислить коэффициенты функциональной зависимости каждого из типов:
        – линейная,
        – параболическая,
        – гиперболическая.
        Известно, что в четвёртой выборке модель логарифмическая, но
        неизвестно основание логарифма.
        - логарифмическая

        :return:
        """


        df = self.data
        results = {}

        for i in range(1, 5):
            x_col = f"X_{i}"
            y_col = f"Y_{i}"
            x_cleaned, y_cleaned = clean_data(df, i)

            if x_col != "X_4" and y_col != "Y_4":

                # Линейная регрессия
                try:
                    b0, b1 = linear_regression(x_cleaned, y_cleaned)
                    results[f"Linear_{i}"] = (b0, b1)
                    print("\n")
                    print(f"Коэффициенты функциональной зависимости для Линейного типа, выборки: {x_col, y_col}")
                    print(f"b0={float(b0)} b1={float(b1)}")
                except Exception as e:
                    print(f"Ошибка в линейной регрессии для {x_col} и {y_col}: {e}")

                # Параболическая регрессия
                try:
                    b0, b1, b2 = parabolic_regression(x_cleaned, y_cleaned)
                    results[f"Parabolic_{i}"] = b0, b1, b2
                    print("\n")
                    print(f"Коэффициенты функциональной зависимости для Параболического типа, выборки: {x_col, y_col}")
                    print(f"b0={b0}, b1={b1}, b2={b2}")
                except Exception as e:
                    print(f"Ошибка в параболической регрессии для {x_col} и {y_col}: {e}")

                # Гиперболическая регрессия
                try:
                    b0, b1 = hyperbolic_regression(x_cleaned, y_cleaned)
                    results[f"Hyperbolic_{i}"] = (b0, b1)
                    print("\n")
                    print(f"Коэффициенты функциональной зависимости для Гиперболического типа, выборки: {x_col, y_col}")
                    print(f"b0={float(b0)} b1={float(b1)}")
                except Exception as e:
                    print(f"Ошибка в гиперболической регрессии для {x_col} и {y_col}: {e}")
            else:

                # Логарифмическая регрессия
                try:

                    b0, b1 = logarithmic_regression(x_cleaned, y_cleaned)
                    results[f"Logarithmic_{i}"] = (b0, b1)
                    print("\n")
                    print(f"Коэффициенты функциональной зависимости для Логарифмического типа, выборки: {x_col, y_col}")
                    print(f"b0={float(b0)} b1={float(b1)}")
                except Exception as e:
                    print(f"Ошибка в логарифмической регрессии для {x_col} и {y_col}: {e}")

        self.coef_func_dependence = results

    def step_3st(self):
        """
        3. Верификация модели. Вычислив коэффициенты детерминации по каждой выборке для каждого исследуемого вида функциональной зависимости,
        выбрать вид зависимости с наибольшим коэффициентом детерминации. Если это линейная модель, то для неё также определить
        коэффициент корреляции и оценить силу и направление линейной зависимости. Для четвёртой выборки подобрать целое значение основания логарифма,
        при котором коэффициент детерминации будет максимальным.

        Примечание 1. Для некоторых моделей возможны очень незначительные (только в 5-6 знаках после запятой) отличия в коэффициентах детерминации.
        Тем не менее должна быть выбрана именно та модель, для которой этот коэффициент больше.

        Примечание 2. При выборе моделей учитывать ОДЗ. Для гиперболической модели при преобразованиях к линейной модели
        и наличии значения аргумента, равного нулю, заменить его значением:
        min_X_k / 1000

        :return:
        """

        df = self.data
        results = {}

        for i in range(1, 5):
            x_cleaned, y_cleaned = clean_data(df, i)

            models = {
                "Linear": linear_regression,
                "Parabolic": parabolic_regression,
                "Hyperbolic": hyperbolic_regression,
                "Logarithmic": logarithmic_regression
            }
            best_model, best_params, best_r_squared = self.select_best_model(x_cleaned, y_cleaned, models)

            if best_model == "Linear":
                evaluate_linear_model(x_cleaned, y_cleaned)
            results[f"Best_Model_{i}"] = (best_model, best_r_squared)

        self.coef_func_dependence = results
        best_model_key = max(results, key=lambda k: results[k][1])
        best_model_name, best_r_squared = results[best_model_key]
        print("\n")
        print(f"Лучшая модель: {best_model_name}, {best_model_key.split('_')[-1]}-й выборке ")

    def step_4st(self):
        """
        4. Построение модели и прогнозирование.
        """
        df = self.data
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        data_pairs = [
            ('X_1', 'Y_1', 'b'),
            ('X_2', 'Y_2', 'g'),
            ('X_3', 'Y_3', 'r'),
            ('X_4', 'Y_4', 'c')
        ]
        for i, (x_col, y_col, color) in enumerate(data_pairs):
            row = i // 2
            col = i % 2
            x = df[x_col].values
            y = df[y_col].values
            x_cleaned, y_cleaned = clean_data(df, i + 1)

            models = {
                "Linear": linear_regression,
                "Parabolic": parabolic_regression,
                "Hyperbolic": hyperbolic_regression,
                "Logarithmic": logarithmic_regression
            }
            best_model, best_params, best_r_squared = self.select_best_model(x_cleaned, y_cleaned, models)

            if best_model == "Linear":
                b0, b1 = best_params
                x_pred = np.linspace(0, max(x) * 3, 100)
                y_pred = b0 + b1 * x_pred
            elif best_model == "Parabolic":
                b0, b1, b2 = best_params
                x_pred = np.linspace(0, max(x) * 3, 100)
                y_pred = b0 + b1 * x_pred + b2 * x_pred ** 2
            elif best_model == "Hyperbolic":
                b0, b1 = best_params
                x_pred = np.linspace(0.001, max(x) * 3, 100)  # Избегаем деления на ноль
                y_pred = b0 / x_pred + b1
            elif best_model == "Logarithmic":
                b0, b1 = best_params
                x_pred = np.linspace(0.001, max(x) * 3, 100)  # Избегаем логарифма от неположительных значений
                y_pred = b0 + b1 * np.log(x_pred)

            axs[row, col].plot(x, y, marker='o', linestyle='-', color=color, label=f'{x_col} vs {y_col}')
            axs[row, col].plot(x_pred, y_pred, linestyle='--', color='k', label=f'Прогноз ({best_model})')
            axs[row, col].set_title(str(i + 1))
            axs[row, col].set_xlabel(x_col)
            axs[row, col].set_ylabel(y_col)
            axs[row, col].legend()
            axs[row, col].grid(True)
        plt.tight_layout()
        plt.show()

    def step_5st(self):
        """
        5. Интерпретация модели.
        """
        df = self.data
        results = {}
        for i in range(1, 5):
            x_cleaned, y_cleaned = clean_data(df, i)
            models = {
                "Linear": linear_regression,
                "Parabolic": parabolic_regression,
                "Hyperbolic": hyperbolic_regression,
                "Logarithmic": logarithmic_regression
            }
            best_model, best_params, best_r_squared = self.select_best_model(x_cleaned, y_cleaned, models)
            results[f"Best_Model_{i}"] = (best_model, best_r_squared)

            print(f"\nАнализ выбора модели для выборки {i}:")
            print(f"Лучшая модель: {best_model}, Коэффициент детерминации: {best_r_squared:.4f}")

            for model_name, model_func in models.items():
                if model_name != best_model:
                    try:
                        if model_name == "Logarithmic":
                            best_base, best_r_squared_log = find_best_log_base(x_cleaned, y_cleaned)
                            params = model_func(x_cleaned, y_cleaned, base=best_base)
                            r_squared = best_r_squared_log
                        else:
                            params = model_func(x_cleaned, y_cleaned)
                            if model_name == "Hyperbolic":
                                b0, b1 = params
                                y_pred = b0 / x_cleaned + b1
                            elif model_name == "Parabolic":
                                b0, b1, b2 = params
                                y_pred = b0 + b1 * x_cleaned + b2 * x_cleaned ** 2
                            else:
                                b0, b1 = params
                                y_pred = b0 + b1 * x_cleaned
                            r_squared = calculate_r_squared(y_cleaned, y_pred)

                        delta_r_squared = best_r_squared - r_squared
                        print("\n")
                        print(
                            f"Модель: {model_name}, Коэффициент детерминации: {r_squared:.4f}, Разница с лучшей моделью: {delta_r_squared:.4f}")

                        if delta_r_squared <= 0.2 * best_r_squared:
                            print(
                                f"Выбор модели {model_name} приемлем, так как разница в коэффициенте детерминации не превышает 20% от лучшей модели.")
                        else:
                            print(
                                f"Выбор модели {model_name} не рекомендуется, так как разница в коэффициенте детерминации превышает 20% от лучшей модели.")

                    except Exception as e:
                        print(f"Ошибка в модели {model_name}: {e}")

    def run(self):
        self.step_1st()
        self.step_2st()
        self.step_3st()
        self.step_4st()
        self.step_5st()


