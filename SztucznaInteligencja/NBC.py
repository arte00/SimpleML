import numpy as np
import sklearn.model_selection

data = np.genfromtxt("wine.data", delimiter=",")

# 6.1

x = data[:, 1:]
y = data[:, 0]

est = sklearn.preprocessing.KBinsDiscretizer(n_bins=3, encode="ordinal", strategy="uniform")
x_disc = est.fit(x)
xt_disc = est.transform(x)

x_train, x_test, y_train, y_test, = sklearn.model_selection.train_test_split(xt_disc, y, test_size=0.33, random_state=42)

counters = []
prob = []
y = []
counters_y = [0, 0, 0]
prob_y = []

for i in range(len(x_train[0])):
    counters_row = [0 for _ in range(9)]
    for k in range(3):
        for j in range(len(x_train)):
            if x_train[j, i] == k and y_train[j] == 1:
                counters_row[k*3] += 1
            elif x_train[j, i] == k and y_train[j] == 2:
                counters_row[k*3+1] += 1
            elif x_train[j, i] == k and y_train[j] == 3:
                counters_row[k*3+2] += 1
    counters.append(counters_row)

for i in y_train:
    if i == 1:
        counters_y[0] += 1
    elif i == 2:
        counters_y[1] += 1
    elif i == 3:
        counters_y[2] += 1

for i in counters_y:
    prob_y.append(i / 119)

for i in counters:
    row = [float(e) for e in i]
    for j in range(9):
        row[j] = (row[j] + 1) / (counters_y[j % 3] + 3)
    prob.append(row)


def predict(x_predict):
    sums = [0, 0, 0]
    for i in range(3):
        sum_y = 1
        for j in range(len(x_predict)):
            value = int(x_predict[j])
            likelihood = prob[j][3*value+i]
            sum_y *= likelihood
        sum_y *= prob_y[i]
        sums[i] = sum_y
    return np.argmax(sums)+1


def show(predict_vector, y_test):
    print(f"predict = {predict_vector}, y={y_test}")


counter = 0
for sample in range(len(y_test)):
    show(predict(x_test[sample]), y_test[sample])
    if predict(x_test[sample]) == int(y_test[sample]):
        counter += 1


print(f"accuracy: {counter / len(y_test)}")

# print(prob[0][6]*prob[1][3]*prob[2][3]*prob[3][3]*prob[4][3]*prob[5][3]*prob[6][3]*prob[7][0]*prob[8][0]*prob[9][3]*prob[10][0]*prob[11][6]*prob[12][3]*prob_y[0])
# print(prob[0][7]*prob[1][4]*prob[2][4]*prob[3][4]*prob[4][4]*prob[5][4]*prob[6][4]*prob[7][1]*prob[8][1]*prob[9][4]*prob[10][1]*prob[11][7]*prob[12][4]*prob_y[1])
# print(prob[0][8]*prob[1][5]*prob[2][5]*prob[3][5]*prob[4][5]*prob[5][5]*prob[6][2]*prob[7][2]*prob[8][2]*prob[9][5]*prob[10][2]*prob[11][8]*prob[12][5]*prob_y[2])















