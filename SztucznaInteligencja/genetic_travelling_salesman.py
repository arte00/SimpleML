import numpy as np
import matplotlib.pyplot as plt
import math


def generate_travelling_salesman_problem(n):
    points = []
    paths = []
    for i in range(n):
        point_x = np.random.rand()
        point_y = np.random.rand()
        points.append([point_x, point_y])
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = calc_distance(points[i], points[j])
                paths.append([i, j, dist])
                paths.append([j, i, dist])
    return np.array(paths)


def calc_distance(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def initialize(n: int, m: int, distance):
    genes = np.array([np.arange(1, n) for _ in range(m)])
    for gene in genes:
        np.random.shuffle(gene)
    print("genes", genes[0])
    return genes, start_fitness(genes, distance)


def get_distance(a, b, distance):
    return distance[np.where(((distance[:, 0] == int(a)) & (distance[:, 1] == int(b))))][0][2]


def start_fitness(genes, distance):
    fits = []
    for gene in genes:
        fit = 0
        fit += get_distance(0, gene[0], distance)
        fit += get_distance(gene[-1], 0, distance)
        for i in range(len(gene) - 1):
            fit += get_distance(gene[i], gene[i + 1], distance)
        fits.append(fit)
    return np.max(fits)


def fitness(genes, distance, start):
    fits = []
    for gene in genes:
        fit = 0
        fit += get_distance(0, gene[0], distance)
        fit += get_distance(gene[-1], 0, distance)
        for i in range(len(gene)-1):
            fit += get_distance(gene[i], gene[i+1], distance)
        fits.append(fit)
    fits = (1 - np.array(fits) / start)
    return fits


def roulette_selection(genes, v, start):
    n = len(genes)
    genes_fitness = fitness(genes, v, start)
    genes_fitness_sum = np.sum(genes_fitness)
    prob = genes_fitness / genes_fitness_sum

    new_population = np.array([genes[np.random.choice(n, p=prob)] for _ in range(n)])
    print("genes roulette", new_population[0])
    return new_population


def ranking_selection(genes, v, start):
    n = len(genes)
    gene_size = len(genes[0])
    genes_fitness = fitness(genes, v, start).reshape(n, 1)
    genes_fitness_sum = n * (n + 1) / 2

    genes = np.append(genes, genes_fitness, axis=1)
    genes_sorted = genes[genes[:, gene_size].argsort()]
    prob = [float(rank) / genes_fitness_sum for rank, fitness in enumerate(genes_sorted, 1)]

    new_population = np.array([genes_sorted[np.random.choice(n, p=prob)] for _ in range(n)])
    return new_population[:, :gene_size]


def tournament_selection(genes, v, start):
    q = 2
    k = len(genes) // q
    np.random.shuffle(genes)

    for i in range(0, k, 2):
        first = genes[i]
        second = genes[i + 1]

        f1 = fitness([first], v, start)
        f2 = fitness([second], v, start)

        if f1 > f2:
            genes[i + 1] = genes[i]
        else:
            genes[i] = genes[i + 1]

    return genes


def pmx(genes):
    ceiling = len(genes[0])
    np.random.shuffle(genes)

    new_population = []

    for i in range(0, len(genes), 2):
        first_parent = genes[i]
        second_parent = genes[i + 1]
        first_cross_point = np.random.randint(ceiling)
        second_cross_point = np.random.randint(ceiling)

        while first_cross_point == second_cross_point:
            second_cross_point = np.random.randint(ceiling)

        if first_cross_point > second_cross_point:
            temp = first_cross_point
            first_cross_point = second_cross_point
            second_cross_point = temp

        first_child = 0
        second_child = 0

        new_population.append(first_child)
        new_population.append(second_child)

    return np.array(new_population)


def cx(genes, n):
    ceiling = n-1
    np.random.shuffle(genes)

    new_population = []

    for i in range(0, len(genes), 2):

        if np.random.rand() < 0.75:
            new_population.append(genes[i])
            new_population.append(genes[i+1])
            continue

        first_parent = genes[i]
        second_parent = genes[i + 1]
        first_cross_point = np.random.randint(ceiling)
        second_cross_point = np.random.randint(ceiling)

        while first_cross_point == second_cross_point:
            second_cross_point = np.random.randint(ceiling)

        if first_cross_point > second_cross_point:
            temp = first_cross_point
            first_cross_point = second_cross_point
            second_cross_point = temp

        '''START'''

        first_child = [0 for _ in range(ceiling)]
        second_child = [0 for _ in range(ceiling)]

        for j in range(first_cross_point, second_cross_point):
            first_child[j] = first_parent[j]
            second_child[j] = second_parent[j]

        copy_len = second_cross_point - first_cross_point

        first_rest = [0 for _ in range(ceiling - copy_len)]
        second_rest = [0 for _ in range(ceiling - copy_len)]

        rest_index = 0
        for j in range(second_cross_point, ceiling):
            first_rest[rest_index] = first_parent[j]
            second_rest[rest_index] = second_parent[j]
            rest_index += 1

        for j in range(0, first_cross_point):
            first_rest[rest_index] = first_parent[j]
            second_rest[rest_index] = second_parent[j]
            rest_index += 1

        copy_index = 0
        for j in range(0, first_cross_point):
            first_child[j] = first_rest[copy_index]
            second_child[j] = second_rest[copy_index]
            copy_index += 1

        for j in range(second_cross_point, ceiling):
            first_child[j] = first_rest[copy_index]
            second_child[j] = second_rest[copy_index]
            copy_index += 1

        new_population.append(first_child)
        new_population.append(second_child)

    return np.array(new_population)


def ox(genes, n):
    ceiling = n - 1
    np.random.shuffle(genes)

    new_population = []

    for i in range(0, len(genes), 2):

        if np.random.rand() < 0.75:
            new_population.append(genes[i])
            new_population.append(genes[i + 1])
            continue

        first_parent = genes[i]
        second_parent = genes[i + 1]

        first_cross_point = np.random.randint(ceiling)
        second_cross_point = np.random.randint(ceiling)

        while first_cross_point == second_cross_point:
            second_cross_point = np.random.randint(ceiling)

        if first_cross_point > second_cross_point:
            temp = first_cross_point
            first_cross_point = second_cross_point
            second_cross_point = temp

        '''START'''

        first_child = [0 for _ in range(len(genes[i]))]
        second_child = [0 for _ in range(len(genes[i+1]))]

        part_one = first_parent[first_cross_point:second_cross_point]
        part_two = second_parent[first_cross_point:second_cross_point]

        rest_one = np.concatenate([first_parent[second_cross_point:], first_parent[:first_cross_point]])
        rest_two = np.concatenate([second_parent[second_cross_point:], first_parent[:first_cross_point]])

        print("rest one", rest_one)
        print("rest two", rest_two)
        print("first parent", first_parent)
        print("second parent", second_parent)

        print("indx", first_cross_point, second_cross_point)
        print("part one", part_one)
        print("part two", part_two)

        copy_len = second_cross_point - first_cross_point

        copy_index = 0
        copy_index1 = 0

        first_child[first_cross_point:second_cross_point] = part_one
        second_child[first_cross_point:second_cross_point] = part_two

        new_population.append(first_child)
        new_population.append(second_child)
        print("first child", first_child)
        print("second child", second_child, "\n")

    return np.array(new_population)


def mutation(genes, n):
    ceiling = n-1
    for gene in genes:
        if np.random.rand() < 0.002:
            first_cross_point = np.random.randint(ceiling)
            second_cross_point = np.random.randint(ceiling)

            while first_cross_point == second_cross_point:
                second_cross_point = np.random.randint(ceiling)

            if first_cross_point > second_cross_point:
                temp = first_cross_point
                first_cross_point = second_cross_point
                second_cross_point = temp

            gene[first_cross_point:second_cross_point] = gene[first_cross_point:second_cross_point][::-1]
    return genes


def show_plot(fit_history, top_values, top_value):
    plt.plot(fit_history)
    plt.scatter([i for i in range(0, len(top_values))], top_values)
    plt.scatter(top_value[0], top_value[1])
    plt.show()


def main():
    # dist_list = np.array([[0, 1, 3.1623], [0, 2, 4.1231], [0, 3, 5.8310], [0, 4, 4.2426],
    #                       [0, 5, 5.3852], [0, 6, 4.0000], [0, 7, 2.2361], [1, 2, 1.0000],
    #                       [1, 3, 2.8284], [1, 4, 2.0000], [1, 5, 4.1231], [1, 6, 4.2426],
    #                       [1, 7, 2.2361], [2, 3, 2.2361], [2, 4, 2.2361], [2, 5, 4.4721],
    #                       [2, 6, 5.0000], [2, 7, 3.1623], [3, 4, 2.0000], [3, 5, 3.6056],
    #                       [3, 6, 5.0990], [3, 7, 4.1231], [4, 5, 2.2361], [4, 6, 3.1623],
    #                       [4, 7, 2.2361], [5, 6, 2.2361], [5, 7, 3.1623], [6, 7, 2.2361],
    #
    #                       [1, 0, 3.1623], [2, 0, 4.1231], [3, 0, 5.8310], [4, 0, 4.2426],
    #                       [5, 0, 5.3852], [6, 0, 4.0000], [7, 0, 2.2361], [2, 1, 1.0000],
    #                       [3, 1, 2.8284], [4, 1, 2.0000], [5, 1, 4.1231], [6, 1, 4.2426],
    #                       [7, 1, 2.2361], [3, 2, 2.2361], [4, 2, 2.2361], [5, 2, 4.4721],
    #                       [6, 2, 5.0000], [7, 2, 3.1623], [4, 3, 2.0000], [5, 3, 3.6056],
    #                       [6, 3, 5.0990], [7, 3, 4.1231], [5, 4, 2.2361], [6, 4, 3.1623],
    #                       [7, 4, 2.2361], [6, 5, 2.2361], [7, 5, 3.1623], [7, 6, 2.2361]
    #                       ])

    n = 20
    m = 10
    iterations = 30

    dist_list = generate_travelling_salesman_problem(n)
    genes, start = initialize(n, m, dist_list)
    fit_history = []
    top_values = []
    top_value = [0, 0]

    for i in range(5):
        genes = roulette_selection(genes, dist_list, start)
        genes = mutation(genes, n)
        fit = fitness(genes, dist_list, start)
        fit_history.append(np.sum(fit) / m)
        top_values.append(np.max(fit))
        if top_values[i] > top_value[1]:
            top_value[0] = i
            top_value[1] = top_values[i]

    show_plot(fit_history, top_values, top_value)

    # print("best result ever: ", top_value[1])


if __name__ == '__main__':
    main()
