from sklearn import datasets
import matplotlib.pyplot as plt

learning_rate = 0.2

m = 100
X, y = datasets.make_blobs(n_samples=m, n_features=2,
                           centers=2, cluster_std=1.05,
                           random_state=2)

x = [[1, X[i, 0], X[i, 1], y[i]] for i in range(m)]


def show_plot(X, y):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Random Classification Data with 2 classes')
    plt.show()


if __name__ == "__main__":
    print("xd")

