items = [[23, 4, "ring"], [18, 3, "ring"], [8, 2, "ring"], [200, 30, "ring"],
         [8, 150, "pen"], [30, 350, "pen"], [5, 100, "pen"], [10, 200, "pen"]]

x_weight = -2
y_weight = 0.1
constant = 6
learn_const = 0.2

ceiling = 8


def network_cell(x: int, y: int) -> str:
    _ratio = (x_weight * x + y_weight * y) + constant
    if _ratio > 0:
        return "ring"
    else:
        return "pen"


def learn(_item: list):
    global x_weight, y_weight, constant
    if _item[2] == "ring":
        x_weight += _item[0] * learn_const
        y_weight += _item[1] * learn_const
        constant += 1 * learn_const
    else:
        x_weight += -1 * _item[0] * learn_const
        y_weight += -1 * _item[1] * learn_const
        constant += -1 * learn_const


def evaluate() -> list:
    evaluation = []
    for item in items:
        result = network_cell(item[0], item[1])
        print(result + "? " + item[2])
        if result == item[2]:
            evaluation.append(True)
        else:
            evaluation.append(False)
    return evaluation


def ratio(evaluations: list):
    counter = 0
    for ev in evaluations:
        if ev:
            counter += 1
    return counter


def check():
    loops = 0
    while ratio(evaluate()) < ceiling:
        results = evaluate()
        index = 0
        for ev in results:
            if ev and len(results) > 1:
                index += 1
            else:
                break
        loops += 1
        print("index: " + str(index))
        learn(items[index])
    print(evaluate())


if __name__ == "__main__":
    check()
    print(ratio(evaluate()))

    new_data = [[40, 10, "ring"]]
    items = new_data
    ceiling = 1
    check()

