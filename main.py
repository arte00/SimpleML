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
    item = []
    x1 = PenOrRing(item, params, 0.2)



