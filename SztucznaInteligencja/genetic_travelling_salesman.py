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
    fits[fits < 0] = 0
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


def pmx(genes, n):
    ceiling = len(genes[0])
    np.random.shuffle(genes)

    new_population = []

    for i in range(0, len(genes), 2):

        first_parent = [i for i in genes[i]]
        second_parent = [i for i in genes[i + 1]]

        if np.random.rand() < 0.75:
            new_population.append(genes[i])
            new_population.append(genes[i + 1])
            continue

        first_cross_point = np.random.randint(ceiling)
        second_cross_point = np.random.randint(ceiling)

        while first_cross_point == second_cross_point:
            second_cross_point = np.random.randint(ceiling)

        if first_cross_point > second_cross_point:
            temp = first_cross_point
            first_cross_point = second_cross_point
            second_cross_point = temp

        child1 = [0 for _ in range(len(first_parent))]
        child2 = [0 for _ in range(len(second_parent))]

        child1[first_cross_point:second_cross_point] = first_parent[first_cross_point:second_cross_point]
        child2[first_cross_point:second_cross_point] = second_parent[first_cross_point:second_cross_point]

        '''CHILD ONE'''

        for idx, x in enumerate(second_parent[first_cross_point:second_cross_point]):
            idx += first_cross_point
            if x not in child1:
                while child1[idx] != 0:
                    idx = second_parent.index(first_parent[idx])
                child1[idx] = x

        for idx in range(len(child1)):
            if child1[idx] == 0:
                child1[idx] = second_parent[idx]

        '''CHILD TWO'''

        for idx, x in enumerate(first_parent[first_cross_point:second_cross_point]):
            idx += first_cross_point
            if x not in child2:
                while child2[idx] != 0:
                    idx = first_parent.index(second_parent[idx])
                child2[idx] = x

        for idx in range(len(child2)):
            if child2[idx] == 0:
                child2[idx] = first_parent[idx]

        new_population.append(child1)
        new_population.append(child2)

    return np.array(new_population)


def cx(genes, n):

    np.random.shuffle(genes)

    new_population = []

    for i in range(0, len(genes), 2):

        if np.random.rand() < 0.75:
            new_population.append(genes[i])
            new_population.append(genes[i + 1])
            continue

        first_parent = genes[i]
        second_parent = genes[i + 1]

        '''START'''

        first_child, second_child = cx_helper(first_parent, second_parent)
        new_population.append(first_child)
        new_population.append(second_child)

    return np.array(new_population)


def cx_helper(parent1_atr, parent2_atr):

    parent1 = [i for i in parent1_atr]
    parent2 = [i for i in parent2_atr]

    curr_index = 0
    child_one = [0 for _ in range(len(parent1))]
    child_two = [0 for _ in range(len(parent2))]

    visited = []
    while curr_index not in visited:
        visited.append(curr_index)
        curr_index = parent1.index(parent2[curr_index])

    for idx in visited:
        child_one[idx] = parent1[idx]

    for idx in range(len(child_one)):
        if child_one[idx] == 0:
            child_one[idx] = parent2[idx]

    curr_index = 0
    visited = []
    while curr_index not in visited:
        visited.append(curr_index)
        curr_index = parent2.index(parent1[curr_index])

    for idx in visited:
        child_two[idx] = parent2[idx]

    for idx in range(len(child_one)):
        if child_two[idx] == 0:
            child_two[idx] = parent1[idx]
    # print(child_one)
    # print(child_two)
    return child_one, child_two


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

        if np.random.rand() < 0.001:
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


def show_plot(fit_history, top_values, top_value, title):
    plt.plot(fit_history)
    plt.scatter([i for i in range(0, len(top_values))], top_values)
    plt.scatter(top_value[0], top_value[1])
    plt.title(title)
    plt.show()


def genetic(dist_list, n, m, it, selection, cross):

    genes, start = initialize(n, m, dist_list)
    fit_history = []
    top_values = []
    top_value = [0, 0]

    for i in range(it):
        if selection == 'roulette':
            genes = roulette_selection(genes, dist_list, start)
        elif selection == 'ranking':
            genes = ranking_selection(genes, dist_list, start)
        elif selection == 'tournament':
            genes = tournament_selection(genes, dist_list, start)

        if cross == 'ox':
            genes = ox(genes, n)
        elif cross == 'cx':
            genes = cx(genes, n)
        elif cross == 'pmx':
            genes = pmx(genes, n)

        genes = mutation(genes, n)

        fit = fitness(genes, dist_list, start)
        fit_history.append(np.sum(fit) / m)
        top_values.append(np.max(fit))

        if top_values[i] > top_value[1]:
            top_value[0] = i
            top_value[1] = top_values[i]

    return fit_history, top_values, top_value


def main():

    selection = ['roulette', 'tournament', 'ranking']
    cross = ['pmx', 'ox', 'cx']
    n = 15
    m = 500
    it = 50
    dist_list = generate_travelling_salesman_problem(n)
    fig, axs = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            history, top_values, top_value = genetic(dist_list, n, m, it, selection[i], cross[j])
            print("top result(" + selection[i] + " + " + cross[j] + "): ", top_value[1])
            axs[i, j].plot(history)
            axs[i, j].scatter([i for i in range(0, len(top_values))], top_values)
            axs[i, j].scatter(top_value[0], top_value[1])
            axs[i, j].set_title(selection[i] + " + " + cross[j])
    plt.show()


if __name__ == '__main__':
    main()
