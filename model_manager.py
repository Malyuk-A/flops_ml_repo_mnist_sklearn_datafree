import warnings
from typing import Any, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from data_manager import DataManager
from flops_utils.ml_repo_templates import ModelManagerTemplate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

mlflow.sklearn.autolog()


class ModelManager(ModelManagerTemplate):
    def __init__(self):
        self.model = LogisticRegression(
            penalty="l2",
            max_iter=1,  # local epoch
            warm_start=True,  # prevent refreshing weights when fitting
        )
        # Setting initial parameters, akin to model.compile for keras models
        self._set_init_params()

    def _set_init_params(self) -> None:
        """Sets initial parameters as zeros Required since model params are uninitialized
        until model.fit is called.

        But server asks for initial parameters from clients at launch. Refer to
        sklearn.linear_model.LogisticRegression documentation for more information.

        Reference: https://github.com/adap/flower/blob/main/examples/sklearn-logreg-mnist/utils.py
        """
        n_classes = 10  # MNIST has 10 classes
        n_features = 784  # Number of features in dataset
        self.model.classes_ = np.array([i for i in range(10)])
        self.model.coef_ = np.zeros((n_classes, n_features))
        if self.model.fit_intercept:
            self.model.intercept_ = np.zeros((n_classes,))

    def set_model_data(self) -> None:
        (self.x_train, self.x_test), (self.y_train, self.y_test) = (
            DataManager().get_data()
        )

    def get_model(self) -> Any:
        return self.model

    def get_model_parameters(self) -> Any:
        if self.model.fit_intercept:
            params = [
                self.model.coef_,
                self.model.intercept_,
            ]
        else:
            params = [
                self.model.coef_,
            ]
        return params

    def set_model_parameters(self, parameters) -> None:
        self.model.coef_ = parameters[0]
        if self.model.fit_intercept:
            self.model.intercept_ = parameters[1]

    def fit_model(self) -> int:
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(self.x_train, self.y_train)
        return len(self.x_train)

    def evaluate_model(self) -> Tuple[Any, Any, int]:
        loss = log_loss(self.y_test, self.model.predict_proba(self.x_test))
        accuracy = self.model.score(self.x_test, self.y_test)
        # return loss, len(self.x_test), {"accuracy": accuracy}
        return loss, accuracy, len(self.x_test)
