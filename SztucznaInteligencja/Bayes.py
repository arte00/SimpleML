import numpy as np
import sklearn.model_selection
import pandas as pd


class BayesDiscrete:

    def __init__(self, x_train, y_train, x_classes, y_classes):

        self.x_classes = x_classes
        self.y_classes = y_classes

        self.x_train = x_train
        self.y_train = y_train
        self.counters = []
        self.prob = []
        self.counters_y = [0 for _ in range(self.y_classes)]
        self.prob_y = []

    def fit(self):
        for i in range(len(self.x_train[0])):
            counters_row = [0 for _ in range(self.x_classes * self.y_classes)]
            for k in range(self.x_classes):
                for j in range(len(self.x_train)):
                    for m in range(self.y_classes):
                        if self.x_train[j, i] == k and self.y_train[j] == m+1:
                            counters_row[k * self.y_classes + m] += 1
            self.counters.append(counters_row)

        for i in self.y_train:
            self.counters_y[int(i) - 1] += 1

        for i in self.counters_y:
            self.prob_y.append(i / len(self.y_train))

        for i in self.counters:
            row = [float(e) for e in i]
            for j in range(self.x_classes * self.y_classes):
                row[j] = (row[j] + 1) / (self.counters_y[j % self.y_classes] + self.y_classes)
            self.prob.append(row)

    def predict(self, x_predict):
        sums = [0 for _ in range(self.y_classes)]
        for i in range(self.y_classes):
            sum_y = 1
            for j in range(len(x_predict)):
                value = int(x_predict[j])
                likelihood = self.prob[j][self.y_classes * value + i]
                sum_y *= likelihood
            sum_y *= self.prob_y[i]
            sums[i] = sum_y
        return np.argmax(sums) + 1
        # return sums


def show(y_predict, y_test):
    print(f"predict = {y_predict}, y={y_test}")


def main():
    data = np.genfromtxt("wine.data", delimiter=",")

    # 6.1

    x = data[:, 1:]
    y = data[:, 0]

    classes = 3

    est = sklearn.preprocessing.KBinsDiscretizer(n_bins=classes, encode="ordinal", strategy="uniform")
    x_disc = est.fit(x)
    xt_disc = est.transform(x)

    x_train, x_test, y_train, y_test, = sklearn.model_selection.train_test_split(
        xt_disc, y, test_size=0.5, random_state=42)

    nbc = BayesDiscrete(x_train, y_train, classes, 3)
    nbc.fit()

    print(f'number of classes: {classes}')

    counter = 0
    for sample in range(len(y_test)):
        result = nbc.predict(x_test[sample])
        show(result, y_test[sample])
        if result == int(y_test[sample]):
            counter += 1

    print(f"accuracy: {counter / len(y_test)}")


def main1():
    data = np.genfromtxt("default of credit card clients.csv", delimiter=";")

    # 6.1

    x = data[2:200, 1:]
    y = data[2:200, -1]

    classes = 200

    est = sklearn.preprocessing.KBinsDiscretizer(n_bins=classes, encode="ordinal", strategy="uniform")
    x_disc = est.fit(x)
    xt_disc = est.transform(x)

    x_train, x_test, y_train, y_test, = sklearn.model_selection.train_test_split(xt_disc, y, test_size=0.3,
                                                                                 random_state=42)
    nbc = BayesDiscrete(x_train, y_train, classes, 2)
    nbc.fit()

    counter = 0
    for sample in range(len(y_test)):
        result = nbc.predict(x_test[sample])
        show(result, y_test[sample])
        if result == int(y_test[sample]):
            counter += 1

    print(f"accuracy: {counter / len(y_test)}")

    print(x_train[0])


if __name__ == '__main__':
    main()


