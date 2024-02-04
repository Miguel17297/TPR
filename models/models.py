from abc import ABC, abstractmethod
from sklearn import svm
from sklearn.ensemble import IsolationForest as IForest
from models.utils import validate_model, distance
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

    def predict(self, test):
        return self.model.predict(test)


    @abstractmethod
    def hyper_tunning(self,train, test, test_labels):
        pass

class OneClassSVM(Model):

    def __init__(self, kernel, nu=0.5):
        model = svm.OneClassSVM(gamma='scale', kernel=kernel, nu=nu)
        super().__init__(model)
        self.__kernel = kernel
        self.__nu = 0.5



    @property
    def kernel(self):
        return self.__kernel

    @property
    def nu(self):
        return self.__nu

    def train(self, data):
        self.model.fit(data)


    def hyper_tunning(self, train, test, test_labels, nu):
        best_score = 0
        best_nu = self.nu
        model = self.model
        for v in nu:
            print(f'\nOneClassSvm {self.kernel} with nu={v}')
            m = svm.OneClassSVM(gamma='scale', kernel=self.kernel, nu=v)
            m.fit(train)
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
        model = IForest(max_samples=max_samples, random_state=random_state)
        super().__init__(model)
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
            print(f'\nIsolation Forest with max_samples={ms} and random_state={rs}')

            clf = IForest(max_samples=ms, random_state=rs)
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

        print(f'Best f1-score: {best_score} -> Isolation Forest with max_samples={self.max_samples} and random_state={self.random_state} \n')

class StatisticalModel(Model):

    def __init__(self, threshold=5):
        self.__threshold = threshold
        self.__centroids = None

    @property
    def threshold(self):
        return self.__threshold

    @property
    def centroids(self):
        return self.__centroids

    def train(self, data):
        self.__centroids = np.mean(data, axis=0)

    def predict(self, test):
        n_obs, _ = test.shape
        results = np.ones(n_obs)
        for i in range(n_obs):
            x = test[i]
            dists = [distance(x, self.centroids)]
            if min(dists) > self.threshold:
                results[i] = -1
            else:
                results[i] = 1

        return results

    def hyper_tunning(self, train, test, test_labels, thresholds):

        if self.centroids is None:
            self.train(train)

        best_score = 0
        best_threshold = self.threshold

        print('\n-- Anomaly Detection based on Centroids Distances --')
        for t in thresholds:
            print(f'Anomaly Threshold: {t}')
            n_obs, _ = test.shape
            results = np.ones(n_obs)
            for i in range(n_obs):
                x = test[i]
                dists = [distance(x, self.centroids)]
                if min(dists) > t:
                    results[i] = -1
                else:
                    results[i] = 1

            score = validate_model(results.reshape(-1,1), test_labels)
            if score > best_score:
                best_score = score
                best_threshold = t

        self.__threshold = best_threshold

        print(
            f'Best f1-score: {best_score} -> Statistical model with threshold={best_threshold} \n')
