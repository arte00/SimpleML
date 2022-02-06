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
        for i in range(len(gene) - 1):
            fit += get_distance(gene[i], gene[i + 1], distance)
        fits.append(fit)
    fits = (1 - np.array(fits) / start)
    return fits


def roulette_selection(genes, v, start):
    n = len(genes)
    genes_fitness = fitness(genes, v, start)
    genes_fitness_sum = np.sum(genes_fitness)
    prob = genes_fitness / genes_fitness_sum

    new_population = np.array([genes[np.random.choice(n, p=prob)] for _ in range(n)])
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


def ox2(genes, n):
    ceiling = len(genes - 1)
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
        second_child = [0 for _ in range(len(genes[i + 1]))]

        part_one = first_parent[first_cross_point:second_cross_point]
        part_two = second_parent[first_cross_point:second_cross_point]

        rest_one = np.concatenate([first_parent[second_cross_point:], first_parent[:second_cross_point]])
        rest_two = np.concatenate([second_parent[second_cross_point:], first_parent[:second_cross_point]])

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

        fixed_pos = list(range(first_cross_point, second_cross_point))

        j = 0
        while j < ceiling:
            if j in fixed_pos:
                j += 1
                continue
            first_child = first_parent

        print("fixed_pos", fixed_pos)

        new_population.append(first_child)
        new_population.append(second_child)
        print("first child", first_child)
        print("second child", second_child, "\n")

    return genes


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
        second_child = [0 for _ in range(len(genes[i + 1]))]

        part_one = first_parent[first_cross_point:second_cross_point]
        part_two = second_parent[first_cross_point:second_cross_point]

        rest_one = np.concatenate([first_parent[second_cross_point:], first_parent[:second_cross_point]])
        rest_two = np.concatenate([second_parent[second_cross_point:], second_parent[:second_cross_point]])

        copy_index = 0
        copy_index1 = 0

        for j in range(second_cross_point, ceiling):
            while rest_two[copy_index] in part_one:
                copy_index += 1
            first_child[j] = rest_two[copy_index]
            copy_index += 1

        for j in range(0, first_cross_point):
            while rest_two[copy_index] in part_one:
                copy_index += 1
            first_child[j] = rest_two[copy_index]
            copy_index += 1

        for j in range(second_cross_point, ceiling):
            while rest_one[copy_index1] in part_two:
                copy_index1 += 1
            second_child[j] = rest_one[copy_index1]
            copy_index1 += 1

        for j in range(0, first_cross_point):
            while rest_one[copy_index1] in part_two:
                copy_index1 += 1
            second_child[j] = rest_one[copy_index1]
            copy_index1 += 1

        first_child[first_cross_point:second_cross_point] = part_one
        second_child[first_cross_point:second_cross_point] = part_two

        new_population.append(first_child)
        new_population.append(second_child)

    return new_population


def mutation(genes, n):
    ceiling = n - 1

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

    n = 30
    m = 200
    iterations = 50

    dist_list = generate_travelling_salesman_problem(n)
    genes, start = initialize(n, m, dist_list)
    fit_history = []
    top_values = []
    top_value = [0, 0]

    for i in range(iterations):
        genes = roulette_selection(genes, dist_list, start)
        genes = mutation(genes, n)
        genes = ox(genes, n)
        fit = fitness(genes, dist_list, start)
        fit_history.append(np.sum(fit) / m)
        top_values.append(np.max(fit))
        if top_values[i] > top_value[1]:
            top_value[0] = i
            top_value[1] = top_values[i]

    show_plot(fit_history, top_values, top_value)
    print("best result ever: ", top_value[1])


if __name__ == '__main__':
    main()
