import numpy as np
import random
from math import sqrt, pi, exp

def gaussian_prob(obs, mu, sig):
    # Calculate Gaussian probability given
    # - observation
    # - mean
    # - standard deviation
    num = (obs - mu) ** 2
    denum = 2 * sig ** 2
    norm = 1 / sqrt(2 * pi * sig ** 2)
    return norm * exp(-num / denum)


def normalize(data):
    return (data[0] - data[1][0]) / (data[1][1] - data[1][0])


def standardize(data):
    return (data[0] - data[1][0]) / data[1][1]


# Gaussian Naive Bayes class
class GNB():
    # Initialize classification categories
    def __init__(self):
        self.classes = ['left', 'keep', 'right']
        self.mean = {}
        self.std = {}

    # Given a set of variables, preprocess them for feature engineering.
    def process_vars(self, vars):
        # The following implementation simply extracts the four raw values
        # given by the input data, i.e. s, d, s_dot, and d_dot.

        # min-max normalize
        # var_min_max = [[-1., 50.], [-2., 10.], [5., 15.], [-3., 3.]]
        # return list(map(normalize, list(zip(vars, var_min_max))))

        # mean-std standardize
        var_mean_std = [[19.81, 11.74], [3.88, 2.96], [9.94, 1.0], [-0.01, 0.9]]
        return list(map(standardize, list(zip(vars, var_mean_std))))

        # return vars

    # Train the GNB using a combination of X and Y, where
    # X denotes the observations (here we have four variables for each) and
    # Y denotes the corresponding labels ("left", "keep", "right").
    def train(self, X, Y):
        '''
        Collect the data and calculate mean and standard variation
        for each class. Record them for later use in prediction.
        '''
        data = {'left': [[] for _ in range(4)], 'keep': [[] for _ in range(4)], 'right': [[] for _ in range(4)]}
        for x, y in zip(X, Y):
            x = self.process_vars(x)
            for i, val in enumerate(x):
                data[y][i].append(val)

        for _class in self.classes:
            tmp_mean, tmp_std = [], []

            for v in data[_class]:
                tmp_mean.append(np.mean(v, axis=0))
                tmp_std.append(np.std(v, axis=0))
            self.mean[_class] = tmp_mean
            self.std[_class] = tmp_std

    # Given an observation (s, s_dot, d, d_dot), predict which behaviour
    # the vehicle is going to take using GNB.
    def predict(self, observation):
        '''
        Calculate Gaussian probability for each variable based on the
        mean and standard deviation calculated in the training process.
        Multiply all the probabilities for variables, and then
        normalize them to get conditional probabilities.
        Return the label for the highest conditional probability.
        '''
        best = -1.
        result = None
        observation = self.process_vars(observation)
        for _class in self.classes:
            prob = 1.
            for o, m, s in zip(observation, self.mean[_class], self.std[_class]):
                prob *= gaussian_prob(o, m, s)
            if prob > best:
                best = prob
                result = _class
        return result
