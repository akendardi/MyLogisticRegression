import random

import numpy as np
import pandas as pd


class MyLogReg:

    def __init__(
            self,
            n_iter: int = 10,  # количество итераций обучения; обычно положительное число
            learning_rate=0.1,  # шаг градиентного спуска; может быть числом или функцией
            metric=None,  # метрика для оценки (accuracy, roc_auc и т.д.)
            reg: str = None,  # тип регуляризации: "l1", "l2", "elasticnet" или None
            l1_coef: float = 0.0,  # коэффициент L1-регуляризации
            l2_coef: float = 0.0,  # коэффициент L2-регуляризации
            sgd_sample=None,  # размер мини-батча для SGD; None = полный градиент
            random_state=42  # фиксируем сид для воспроизводимости случайных выборок
    ):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None # веса инициализированы пустым массивом; позже нужно заполнить нулями/единицами
        self.metric = metric
        self.score = 0  # хранит значение метрики на полном датасете
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.random_state = random_state
        self.sgd_sample = sgd_sample  # может быть int, float (доля) или None


    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        random.seed(self.random_state)  # фиксируем сид для воспроизводимости случайной подвыборки
        X = self._prepare_x(X)  # добавляем bias (столбец единиц) и преобразуем X к DataFrame при необходимости
        self.weights = np.zeros(len(X.columns))  # инициализация весов; лучше нулями, чтобы не смещать градиент

        for i in range(1, self.n_iter + 1):
            # формируем мини-батч для стохастического градиентного спуска
            sample_row_idx = random.sample(range(X.shape[0]), self._get_sample_size(X))
            X_train = X.iloc[sample_row_idx]  # подвыборка признаков
            y_train = pd.Series(y).iloc[sample_row_idx]  # подвыборка меток

            pred = self.predict_proba(X_train)  # вероятности для выбранного батча (важно: именно сигмоида)
            error = self._get_loss_error(X, y)  # loss вычисляется на всем датасете, а не на батче

            # вывод информации о процессе обучения
            if verbose != False and i % verbose == 0:
                verbose_str = f"{i} | loss: {error}"
                if self.metric != None:
                    verbose_str += f" | {self.metric}: {self.get_metric_value(y, X)}"
                print(verbose_str)

            # learning_rate может быть функцией или числом
            learn_rate = self.learning_rate(i) if callable(self.learning_rate) else self.learning_rate

            # вычисление градиента на батче
            grad = (pred - y_train) @ X_train / len(X_train) + self._get_reg_grad_value()

            # обновление весов
            self.weights = self.weights - learn_rate * grad

            # сохраняем текущую метрику на всем датасете
            self.score = self.get_metric_value(y, X)

    def _get_sample_size(self, X: pd.DataFrame):
        if self.sgd_sample == None:  # если SGD не используется, то берём весь датасет
            return X.shape[0]
        if isinstance(self.sgd_sample, float):  # если задана доля (например, 0.1)
            return round(X.shape[0] * self.sgd_sample)  # переводим её в количество строк (округляем)
        else:  # если задано целое число — значит это количество строк
            return self.sgd_sample

    def _get_loss_error(self, X: pd.DataFrame, y: pd.Series):
        eps = 1 * 10 ** (-15)  # маленькая константа для защиты от log(0)
        pred = self.predict_proba(X)  # предсказываем вероятности
        return -np.sum(
            y * np.log(pred + eps) + (1 - y) * np.log(1 - pred + eps)) / len(X) \
            + self._get_reg_error_value()  # считаем логистическую функцию потерь (binary cross-entropy) + регуляризацию

    def predict_proba(self, X: pd.DataFrame):
        X = self._prepare_x(X)  # добавляем bias (если ты делаешь это в _prepare_x)
        return 1 / (1 + np.exp(-(X @ self.weights)))  # классическая сигмоида: P(y=1 | x)

    def predict(self, X: pd.DataFrame):
        proba = self.predict_proba(X)  # получаем вероятности
        return (proba >= 0.5).astype(int)  # жёсткий порог в 0.5 для классификации

    def get_coef(self):
        return self.weights[1:]

    def get_metric_value(self, y: pd.Series, X: pd.DataFrame):
        if self.metric == "accuracy":
            return self.get_accuracy(y, X)
        if self.metric == "precision":
            return self.get_precision(y, X)
        if self.metric == "recall":
            return self.get_recall(y, X)
        if self.metric == "f1":
            return self.get_f1(y, X)
        if self.metric == "roc_auc":
            return self.get_roc_auc(y, X)

    def _get_reg_grad_value(self):
        if self.reg == "l1":
            return self.l1_coef * np.sign(self.weights[:])
            # L1-регуляризация: градиент — знак весов (субградиент).
        if self.reg == "l2":
            return self.l2_coef * 2 * self.weights[:]
            # L2-регуляризация: градиент — 2λw.
        if self.reg == "elasticnet":
            return self.l1_coef * np.sign(self.weights) + self.l2_coef * 2 * self.weights[:]
            # ElasticNet = L1 + L2 одновременно.
        return 0
        # Возвращает добавку к градиенту для регуляризации.

    def _get_reg_error_value(self):
        if self.reg == "l1":
            return self.l1_coef * np.sum(np.abs(self.weights))
            # L1: штраф = λ * сумма модулей весов.
        if self.reg == "l2":
            return self.l2_coef * np.sum(self.weights ** 2)
            # L2: штраф = λ * ||w||².
        if self.reg == "elasticnet":
            return self.l1_coef * np.sum(np.abs(self.weights)) + self.l2_coef * np.sum(self.weights ** 2)
            # ElasticNet = комбинация штрафов.
        return 0
        # Возвращает добавку к функции ошибки (loss).

    def get_accuracy(self, y: pd.Series, X: pd.DataFrame):
        pred = self.predict(X)
        tp, fn, fp, tn = self._get_confusion_matrix_np(y, pred)
        return (tp + tn) / (tp + fn + fp + tn)

    def get_precision(self, y: pd.Series, X: pd.DataFrame):
        pred = self.predict(X)
        tp, fn, fp, tn = self._get_confusion_matrix_np(y, pred)
        return tp / (tp + fp)

    def get_recall(self, y: pd.Series, X: pd.DataFrame):
        pred = self.predict(X)
        tp, fn, fp, tn = self._get_confusion_matrix_np(y, pred)
        return tp / (tp + fn)

    def get_f1(self, y: pd.Series, X: pd.DataFrame):
        recall = self.get_recall(y, X)
        precision = self.get_precision(y, X)
        return 2 * precision * recall / (precision + recall)

    def get_roc_auc(self, y_true: pd.Series, X: pd.DataFrame) -> float:

        proba = self.predict_proba(X)  # предсказанные вероятности
        y_score = proba.round(10)  # округляем, чтобы избежать проблем с float

        df = pd.DataFrame({"y": y_true, "scores": y_score})
        df = df.sort_values("scores", ascending=False)  # сортируем по предсказанным вероятностям

        positives = df[df["y"] == 1]["scores"].values  # вероятности для положительного класса
        negatives = df[df["y"] == 0]["scores"].values  # вероятности для отрицательного класса

        total = 0
        for neg in negatives:  # перебираем отрицательные примеры
            score_higher = np.sum(positives > neg)  # сколько положительных имеют бОльший скор
            score_equal = np.sum(positives == neg)  # сколько равны (дают 0.5)
            total += score_higher + score_equal / 2

        # нормализуем на число всех пар (positive, negative)
        return total / (len(positives) * len(negatives))

    def _get_confusion_matrix_np(self, y, y_pred):
        y = np.array(y)
        y_pred = np.array(y_pred)

        tp = np.sum((y == 1) & (y_pred == 1))  # истинно-положительные
        fn = np.sum((y == 1) & (y_pred == 0))  # ложно-отрицательные
        fp = np.sum((y == 0) & (y_pred == 1))  # ложно-положительные
        tn = np.sum((y == 0) & (y_pred == 0))  # истинно-отрицательные

        return tp, fn, fp, tn

    def get_best_score(self):
        return self.score

    def _prepare_x(self, X: pd.DataFrame):
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = pd.DataFrame(X)

        X = X.copy()
        if "w0" not in X.columns:
            X.insert(0, "w0", 1)
        return X



