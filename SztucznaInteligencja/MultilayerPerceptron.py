import numpy as np
from sklearn import datasets
import sklearn.model_selection as ms
import matplotlib.pyplot as plt


def fi(s):
    return 1 / (1 + np.exp(-s))


def fit(X, y, n, n_iterations, learning_rate):
    V = (np.random.rand(n, X.shape[1]+1)*2-1)/1000
    W = (np.random.rand(n+1)*2-1)/1000

    for _ in range(n_iterations):
        inx = np.random.randint(X.shape[0])

        x = X[inx]
        s = V[:, 1:].dot(x) + V[:, 0]
        si = fi(s)
        y_output = np.dot(si, W[1:]) + W[0]
        V[:, 1:] -= learning_rate * (y_output - y[inx]) * np.outer(W[1:] * si * (1 - si), x)
        V[:, 0] -= learning_rate * (y_output - y[inx]) * W[1:] * si * (1 - si)
        W[1:] -= learning_rate * (y_output - y[inx]) * si
        W[0] -= learning_rate * (y_output - y[inx])

    return V, W


def output(V, W, X):
    y = []
    for x in X:
        s = V[:, 1:].dot(x) + V[:, 0]
        si = fi(s)
        y_output = np.dot(si, W[1:]) + W[0]
        y.append(y_output)
    return np.array(y)

def main():

    def fi(s):
        return 1 / (1 + np.exp(-s))

    def si(row):
        output = 0
        for idx, j in enumerate(row):
            output += j * X[idx]
        return output

    def calculate_y():
        output = 0
        for idx, j in enumerate(w):
            output += j * fi(si(v[idx]))
        return output

    def correct_v():
        for i in range(n):
            for j in range(m):
                s = fi(si(v[i]))
                v[i][j] -= learning_rate*(y_output - y[i])*w[i]*s*(1-s)*X[j]

    def correct_w():
        for j in range(n):
            s = fi(si(v[j]))
            w[j] -= learning_rate*(y_output - y[j])*s

    m = 100
    n = 3

    X, y = datasets.make_blobs(n_samples=m, n_features=2,
                               centers=2, cluster_std=1.15,
                               random_state=123)
    # fig = plt.figure(figsize=(10, 8))
    # plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
    # plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    # plt.xlabel("feature 1")
    # plt.ylabel("feature 2")
    # plt.title('Random Classification Data with 2 classes')
    # plt.show()

    v = [[0.3 for _ in range(m)] for _ in range(n)]
    w = [0.3 for _ in range(n)]
    y_output = None
    learning_rate = 0.1
    t = 100

    for _ in range(t):
        y_output = calculate_y()
        correct_v()
        correct_w()
    y_output = calculate_y()
    print(w)

    print(y_output)


def main2():
    X = np.random.rand(100, 2) * np.pi

    y = np.cos(X[:, 0] * X[:, 1])*np.cos(X[:, 0]*2)
    v, w = fit(X, y, 20, 1000000, 0.01)

    xx, xy = np.meshgrid(np.linspace(0, np.pi), np.linspace(0, np.pi))
    xz = np.reshape(output(v, w, np.c_[xx.flatten(), xy.flatten()]), xx.shape)

    _, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(xx, xy, xz)
    plt.show()


if __name__ == "__main__":
    main2()

