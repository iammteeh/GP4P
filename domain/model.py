from dataclasses import dataclass, field
from typing import Any
from domain.dataset import DataSet
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, accuracy_score, precision_score, roc_curve
from sklearn.inspection import permutation_importance, plot_partial_dependence
import graphviz
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

    @property
    def program(self):
        return self.data.program
    
    @program.setter
    def program(self, program):
        self.data.program = program

    def eval(self):
            if "mse" in self.metrics and "mse" not in self._evaluation.keys():
                self._evaluation["mse"] = mean_squared_error(self.y_test, self.y_pred)
                self.eval()
            elif "mse_relative_to_mean" in self.metrics and "mse_relative_to_mean" not in self._evaluation.keys():
                self._evaluation["mse_relative_to_mean"] = mean_squared_error(self.y_test, self.y_pred) / np.mean(self.y_test)
                self.eval()
            elif "mse_relative_to_variance" in self.metrics and "mse_relative_to_variance" not in self._evaluation.keys():
                self._evaluation["mse_relative_to_variance"] = mean_squared_error(self.y_test, self.y_pred) / np.var(self.y_test)
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

    def plot_symbolic_program(self, *features):
        dot = self.data.program.export_graphviz()
        # replace variable names with feature names
        for i, feature in enumerate(self.data.get_all_config_df().columns):
            dot = dot.replace(f"X{i}", feature)
        graph = graphviz.Source(dot)
        graph.render('symbolic expression', view=True, cleanup=True)
        # generate partial dependence plot for feature string
        if features:
            features = np.arrange(features.shape[1])
            plot_partial_dependence(self.method, self.X_test, features, feature_names=self.data.columns[:-1])
