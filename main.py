from PenOrRing import *

if __name__ == "__main__":
    items = [[23, 4, "ring"], [18, 3, "ring"], [8, 2, "ring"], [200, 30, "ring"],
             [8, 150, "pen"], [30, 350, "pen"], [5, 100, "pen"], [10, 200, "pen"]]
    x = PenOrRing(items, [-2, 0.1, 6], 0.2)
    print(x.evaluate())
    print(x.ratio())
    x.machine_learning(8)
    print(x.evaluate())
    print(x.ratio())
    params = x.get_params()
    items2 = [[40, 10, "ring"], [0.5, 5, "pen"], [500, 10000, "pen"], [20, 5, "ring"]]
    x1 = PenOrRing(items2, params, 0.2)

    print(x1.network_cell(items2[0]))
    print(x1.network_cell(items2[1]))
    print(x1.network_cell(items2[2]))
    print(x1.network_cell(items2[3]))



