from dataclasses import dataclass, field
from typing import Any
from domain.dataset import DataSet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, roc_curve
import time

@dataclass
class Model:
    method: str
    metrics: list
    data: DataSet
    y_test: Any
    _y_pred: Any = None
    _evaluation: dict[str, float] = field(default_factory=dict)

    @property
    def y_pred(self):
        return self._y_pred
    
    @y_pred.setter
    def y_pred(self, y_pred):
        self._y_pred = y_pred

    @property
    def coef(self):
        return self.data.coef
    
    @coef.setter
    def coef(self, coef):
        self.data.coef = coef

    def eval(self):
            if "mse" in self.metrics and "mse" not in self._evaluation.keys():
                self._evaluation["mse"] = mean_squared_error(self.y_test, self.y_pred)
                self.eval()
            elif "mape" in self.metrics and "mape" not in self._evaluation.keys():
                self._evaluation["mape"] = mean_absolute_percentage_error(self.y_test, self.y_pred)
                self.eval()
            elif "r2" in self.metrics and "r2" not in self._evaluation.keys():
                self._evaluation["r2"] = r2_score(self.y_test, self.y_pred)
                self.eval()
            elif "accuracy" in self.metrics and "accuracy" not in self._evaluation.keys():
                self._evaluation["accuracy"] = accuracy_score(self.y_test, self.y_pred)
                self.eval()
            elif "precision" in self.metrics and "precision" not in self._evaluation.keys():
                self._evaluation["precision"] = precision_score(self.y_test, self.y_pred)
                self.eval()
            elif "roc" in self.metrics and "roc" not in self._evaluation.keys():
                self._evaluation["roc"] = roc_curve(self.y_test, self.y_pred)
                self.eval()
            else:
                return self._evaluation
            
            if sorted(self._evaluation.keys()) == sorted(self.metrics):
                return self._evaluation
            else:
                raise ValueError("Unknown metric")
    
    def eval_alt(self):
        for metric in self.metrics:
            if metric == "mse":
                self._evaluation["mse"] = mean_squared_error(self.y_test, self.y_pred)
            elif metric == "mape":
                self._evaluation["mape"] = mean_absolute_percentage_error(self.y_test, self.y_pred)
            elif metric == "r2":
                self._evaluation["r2"] = r2_score(self.y_test, self.y_pred)
            elif metric == "accuracy":
                self._evaluation["accuracy"] = accuracy_score(self.y_test, self.y_pred)
            elif metric == "precision":
                self._evaluation["precision"] = precision_score(self.y_test, self.y_pred)
            elif metric == "roc":
                self._evaluation["roc"] = roc_curve(self.y_test, self.y_pred)
            else:
                raise ValueError("Unknown metric")
            
    def __str__(self):
        return f"Model: {self.method} with metrics: {self.metrics}"
    
    def __repr__(self):
        return f"Model: {self.method} with metrics: {self.metrics}"
    
    def evaluation(self):
        start = time.time()
        self.eval()
        end = time.time()

        print(f"evaluation took {end - start}")
        print(self._evaluation)
        for eval, value in self._evaluation.items():
            print(f"{eval}: {value}")

    def test_evaluation(self):
        i = 0
        timings_eval_alt = []
        while i < 1000:
            self._evaluation = {}
            start = time.time()
            self.eval_alt()
            end = time.time()
            time_of_evaluation = end - start
            timings_eval_alt.append(time_of_evaluation)
            i += 1
        print(f"evaluation of eval_alt took {sum(timings_eval_alt)} in total and in average {sum(timings_eval_alt) / len(timings_eval_alt)}")
        i = 0
        timings_eval = []
        while i < 1000:
            self._evaluation = {}
            start = time.time()
            self.eval()
            end = time.time()
            time_of_evaluation = end - start
            timings_eval.append(time_of_evaluation)
            i += 1
        print(f"evaluation of eval took {sum(timings_eval)} in total and in average {sum(timings_eval) / len(timings_eval)}")
        