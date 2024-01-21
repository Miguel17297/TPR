from abc import ABC, abstractmethod
from sklearn import svm
from utils import validate_model, distance
import itertools as iter
import numpy as np

class Model(ABC):

    def __init__(self, model):
        self.__model = model

    @property
    def model(self):
        return self.__model

    def train(self, data):
        self.model.fit(data)

    def test(self, test):
        return self.model.predict(test)

    @abstractmethod
    def hyper_tunning(self,train, test, test_labels):
        pass

class OneClassSVM(Model):

    def __init__(self, kernel, nu=0.5):
        self.__kernel = kernel
        self.__nu = 0.5
        model = svm.OneClassSVM(gamma='scale', kernel=kernel, nu=nu)
        super.__init__(model)

    @property
    def kernel(self):
        return self.__kernel

    @property
    def nu(self):
        return self.nu

    def train(self, data):
        self.model.fit(data)

    def test(self, test):
        return self.model.predict(test)

    def hyper_tunning(self, train, test, test_labels, nu):
        best_score = 0
        best_nu = self.nu
        model = self.model
        for v in nu:

            m = svm.OneClassSVM(gamma='scale', kernel='linear', nu=v).fit(train)
            res = m.predict(test)

            score = validate_model(res.reshape(-1, 1), test_labels)

            if score > best_score:
                best_score = score
                best_nu = v
                model = m

        self.__nu = best_nu
        self.__model = model

        print(f'Best f1-score: {best_score} -> OneClassSvm {self.kernel} with nu={self.nu} \n')


class IsolationForest(Model):
    def __init__(self, max_samples=100, random_state=0):
        model = IsolationForest(max_samples=max_samples, random_state=random_state)
        super.__init__(model)
        self.__max_samples = max_samples
        self.__random_state = random_state

    @property
    def max_samples(self):
        return self.__max_samples

    @property
    def random_state(self):
        return self.__random_state

    def hyper_tunning(self, train, test, test_labels, max_samples, random_state):

        best_score = 0
        best_ms = self.max_samples
        best_rs = self.random_state
        model = self.model

        for ms, rs in [*iter.product(max_samples, random_state)]:

            clf = IsolationForest(max_samples=ms, random_state=rs)
            clf.fit(train)
            res= clf.predict(test)
            score = validate_model(res.reshape(-1,1), test_labels)
            if score > best_score:
                best_score = score
                best_ms = ms
                best_rs = rs
                model = clf

        self.__random_state = best_rs
        self.__max_samples = best_ms
        self.__model = model

        print(
            f"Best f1-score: {best_score} -> Isolation Forest with max_samples={self.max_samples} and random_state={self.random_state} \n")


# class StatisticalModel(Model):
#
#     def __init__(self, kernel, nu=0.5):
#         pass
#
#     def train(self, data):
#         centroids = np.mean(data, axis=0)
#
#     def test(self, test):
#         nObsTest, nFea = data.shape
#         results = np.ones(nObsTest)
#         for i in range(nObsTest):
#             x = data[i]
#             dists = [distance(x, centroids)]
#             if min(dists) > j:
#                 results[i] = -1
#             else:
#                 results[i] = 1
#
#     def hyper_tunning(self, train, test, test_labels):
#         pass

