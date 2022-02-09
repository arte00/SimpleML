
def pmx(parent1, parent2, first_cross_point, second_cross_point):

    child1 = [0 for _ in range(len(parent1))]

    child1[first_cross_point:second_cross_point] = parent1[first_cross_point:second_cross_point]

    for idx, x in enumerate(parent2[first_cross_point:second_cross_point]):
        idx += first_cross_point
        if x not in child1:
            while child1[idx] != 0:
                idx = parent2.index(parent1[idx])
            child1[idx] = x

    for idx in range(len(child1)):
        if child1[idx] == 0:
            child1[idx] = parent2[idx]

    return child1


if __name__ == "__main__":
    a = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    c = pmx(a, b, 3, 7)
    d = pmx(b, a, 3, 7)
    print("Parents:")
    print(a)
    print(b)
    print("Children:")
    print(c)
    print(d)
