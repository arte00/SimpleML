import numpy as np
from sklearn import datasets
import sklearn.model_selection as ms
import matplotlib.pyplot as plt


def _unit_step_function(x):
    return np.where(x >= 0, 1, 0)


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.alpha = learning_rate
        self.n_iterations = n_iterations
        self.activation_func = _unit_step_function
        self.weights = None
        self.bias = None
        self.k = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights

        self.weights = np.zeros(n_features)
        self.bias = 0

        _y = np.array([1 if i > 0 else 0 for i in y])

        while self.k < self.n_iterations:
            predict = []
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                predict.append(y_predicted)
                update = self.alpha * (_y[idx] - y_predicted)

                if update != 0:
                    self.weights += update * x_i
                    self.bias += update
                    self.k += 1

            y1 = np.array([np.array(i) for i in _y])
            if (predict == y1).all():
                break

        return self.k, self.weights

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    def show_plot(self, X, y):
        fig = plt.figure(figsize=(10, 8))
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
        plt.xlabel("feature 1")
        plt.ylabel("feature 2")
        plt.title('Random Classification Data with 2 classes')

        x0_1 = np.amin(X[:, 0])
        x0_2 = np.amax(X[:, 0])

        x1_1 = (-self.weights[0] * x0_1 - self.bias) / self.weights[1]
        x1_2 = (-self.weights[0] * x0_2 - self.bias) / self.weights[1]

        plt.plot([x0_1, x0_2], [x1_1, x1_2], 'k')

        plt.show()


def task1():
    m = 133
    X, y = datasets.make_blobs(n_samples=m, n_features=2,
                               centers=2, cluster_std=1.15,
                               random_state=6)

    perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)

    k, weights = perceptron.fit(X, y)

    print("iterations: " + str(k))

    perceptron.show_plot(X, y)


def test_predict():
    m = 100
    X, y = datasets.make_blobs(n_samples=m, n_features=2,
                               centers=2, cluster_std=1.2,
                               random_state=2)
    perceptron = Perceptron(learning_rate=0.1, n_iterations=1000)

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=100)

    k, weights = perceptron.fit(X_train, y_train)
    predicted = perceptron.predict(X_test)

    print("iterations: " + str(k))
    print("accuracy: " + str(accuracy(y_test, predicted)))

    perceptron.show_plot(X, y)


if __name__ == "__main__":
    task1()
