import numpy as np
import matplotlib.pyplot as plt


def generate_knapsack_problem(n):
    w = [10 * np.random.randint(100) for _ in range(n)]
    v = [np.random.randint(100) for _ in range(n)]
    ran_c = np.random.rand()
    while ran_c < 0.5:
        ran_c = np.random.rand()
    c = int(np.sum(w) * ran_c)
    return w, v, c


def knapsack(W, wt, val, n):

    if n == 0 or W == 0:
        return 0

    if wt[n - 1] > W:
        return knapsack(W, wt, val, n - 1)

    else:
        return max(
            val[n - 1] + knapsack(
                W - wt[n - 1], wt, val, n - 1),
            knapsack(W, wt, val, n - 1))


def initialize(n: int, m: int):
    genes = np.array([np.random.randint(2, size=n) for _ in range(m)])
    return genes


def fitness(x, v, c):
    return np.array([fitness_function(gene, v, c) for gene in x])


def fitness_function(x, v, c):
    weight = 0
    for i in range(len(x)):
        weight += v[i][1] * x[i]
    if weight > c:
        return 0
    fitness = 0
    for i in range(len(x)):
        fitness += v[i][0] * x[i]
    return fitness


def fitness_function_heuristic(x, v, c):
    fitness = 0
    for i in range(len(x)):
        fitness += v[i][0] * x[i] - (x[i] * v[i][1] - c) ** 2
    return fitness


def roulette_selection(genes, v, c):
    n = len(genes)

    genes_fitness = np.array([fitness_function(gene, v, c) for gene in genes])
    genes_fitness_sum = np.sum(genes_fitness)
    prob = [gene / genes_fitness_sum for gene in genes_fitness]

    new_population = np.array([genes[np.random.choice(n, p=prob)] for _ in range(n)])

    return new_population


def ranking_selection(genes, v, c):
    n = len(genes)
    gene_size = len(genes[0])
    genes_fitness = np.array([fitness_function(gene, v, c) for gene in genes]).reshape(n, 1)
    genes_fitness_sum = n * (n + 1) / 2

    genes = np.append(genes, genes_fitness, axis=1)
    genes_sorted = genes[genes[:, gene_size].argsort()]
    prob = [float(rank) / genes_fitness_sum for rank, fitness in enumerate(genes_sorted, 1)]

    new_population = np.array([genes_sorted[np.random.choice(n, p=prob)] for _ in range(n)])
    return new_population[:, :gene_size]


def tournament_selection(genes, v, c):
    q = 2
    k = len(genes) // q
    np.random.shuffle(genes)

    for i in range(0, k, 2):
        first = genes[i]
        second = genes[i + 1]

        f1 = fitness_function(first, v, c)
        f2 = fitness_function(second, v, c)

        if f1 > f2:
            genes[i + 1] = genes[i]
        else:
            genes[i] = genes[i + 1]

    return genes


def one_point_crossing(genes):
    ceiling = len(genes[0])
    np.random.shuffle(genes)

    new_population = []

    for i in range(0, len(genes), 2):
        first_parent = genes[i]
        second_parent = genes[i + 1]
        cross_point = np.random.randint(ceiling)

        first_child = np.append(np.array([first_parent[0:cross_point]]),
                                np.array(second_parent[cross_point:]))

        second_child = np.append(np.array([second_parent[0:cross_point]]),
                                 np.array(first_parent[cross_point:]))

        new_population.append(first_child)
        new_population.append(second_child)

    return np.array(new_population)


def two_point_crossing(genes):
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

        first_child = np.append(np.array([first_parent[0:first_cross_point]]),
                                np.array([second_parent[first_cross_point:second_cross_point]]))
        first_child = np.append(first_child, np.array([first_parent[second_cross_point:]]))

        second_child = np.append(np.array([second_parent[0:first_cross_point]]),
                                 np.array([first_parent[first_cross_point:second_cross_point]]))

        second_child = np.append(second_child, np.array([second_parent[second_cross_point:]]))

        new_population.append(first_child)
        new_population.append(second_child)

    return np.array(new_population)


def mutation(genes, mutation_prob):
    i_len = len(genes)
    j_len = len(genes[0])
    mut_counter = 0

    for i in range(i_len):
        for j in range(j_len):
            if np.random.rand() < mutation_prob:
                mut_counter += 1

                if genes[i][j] == 1:
                    genes[i][j] = 0
                else:
                    genes[i][j] = 1

    return genes


def genetic_algorithm(V, c, iterations, m, mutation_rate, selection, crossing):
    n = len(V)

    genes = initialize(n, m)

    history = []
    top_genes = []

    history.append(np.sum(fitness(genes, V, c)) / len(genes))
    top_genes.append(np.max(fitness(genes, V, c)))

    for _ in range(iterations):

        '''SELECTION'''

        if selection == "roulette":
            genes = roulette_selection(genes, V, c)
        elif selection == "ranking":
            genes = ranking_selection(genes, V, c)
        elif selection == "tournament":
            genes = tournament_selection(genes, V, c)

        '''CROSSING'''

        if crossing == "one_point":
            genes = one_point_crossing(genes)
        elif crossing == "two_point":
            genes = two_point_crossing(genes)

        '''MUTATION'''

        genes = mutation(genes, mutation_rate)

        history.append(np.sum(fitness(genes, V, c)) / len(genes))
        top_genes.append(np.max(fitness(genes, V, c)))

    return genes, fitness(genes, V, c), history, top_genes, np.max(top_genes)


def testing():
    values = [
        360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
        78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
        87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
        312
    ]
    weights = [
        7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
        42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
        3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
    ]
    n = len(values)
    m = 100
    c = 850
    V = [[values[i], weights[i]] for i in range(n)]
    genes = initialize(n, m)
    tournament_selection(genes, V, c)

    # for _ in range(25):
    #     g = fitness(genes, V, c)
    #     print(np.sum(g) / len(g))
    #     genes = roulette_selection(genes, V, c)

    # print(len(one_point_crossing(genes)))


def task1():
    # values = [
    #     360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
    #     78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
    #     87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
    #     312
    # ]
    # weights = [
    #     7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    #     42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    #     3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
    # ]
    # n = len(values)
    new_n = 20
    weights, values, c = generate_knapsack_problem(new_n)
    m = 400
    # c = 850
    iterations = 25
    mutation_prob = 0.001
    V = [[values[i], weights[i]] for i in range(new_n)]
    result, fitness, history, top, top_one = genetic_algorithm(V[0:20],
                                                               c, iterations,
                                                               m, mutation_prob,
                                                               "tournament",
                                                               "two_point")
    print(fitness)
    print(history)
    print(top)
    print("best result ever ", top_one)

    solution = knapsack(c, weights[0:new_n], values[0:new_n], new_n)
    print("solution: ", solution)

    plt.plot(history)
    plt.scatter([i for i in range(0, len(top))], top, color='red')
    plt.show()


'''

MAIN

'''

if __name__ == '__main__':
    task1()
