from keras.models import load_model
import numpy as np
import keras.losses

from Utils.utils import get_network_type


class TestCaseSplitter:
    def __init__(self, filename, X_test, y_test, comparator):
        self.__model = load_model(filename)
        self.__X_test = X_test
        self.__y_test = y_test
        self.__comparator = comparator
        self.__passing_test_inputs = []
        self.__passing_test_outputs = []
        self.__failing_test_inputs = []
        self.__failing_test_actual = []
        self.__failing_test_expected = []
        # self.loss_func = self.__model.loss.__name__
        if isinstance(self.__model.loss, keras.losses.Loss):
            self.loss_func = type(self.__model.loss).__name__
        elif callable(self.__model.loss):
            self.loss_func = self.__model.loss.__class__.__name__
        else:
            self.loss_func = str(self.__model.loss)
    def split(self):
        r = self.__model.predict(self.__X_test)
        self.__model.summary()
        for i in range(0, len(r)):
            actual = r[i]
            expected = self.__y_test[i]
            
            if self.__comparator.compare(expected, actual):
                self.__passing_test_inputs.append(np.asarray(self.__X_test[i]))
                self.__passing_test_outputs.append(expected)
            else:
                self.__failing_test_inputs.append(np.asarray(self.__X_test[i]))
                self.__failing_test_actual.append(actual)
                self.__failing_test_expected.append(expected)
    def getModelType(self):
        return get_network_type(self.__model)

    def get_passing_test_inputs(self):
        return self.__passing_test_inputs

    def get_passing_test_outputs(self):
        return self.__passing_test_outputs

    def get_failing_test_inputs(self):
        return self.__failing_test_inputs

    def get_failing_test_actual_outputs(self):
        return self.__failing_test_actual

    def get_failing_test_expected_outputs(self):
        return self.__failing_test_expected
