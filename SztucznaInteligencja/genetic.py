import numpy as np
import matplotlib.pyplot as plt
import knapsack


def generate_knapsack_problem(n):
    w = [10 * np.random.randint(100) for _ in range(n)]
    v = [np.random.randint(100) for _ in range(n)]
    ran_c = np.random.rand()
    while ran_c < 0.5:
        ran_c = np.random.rand()
    c = int(np.sum(w) * ran_c)
    return w, v, c

def generate_travelling_salesman_problem(n):
    pass


def knapsack_force(W, wt, val, n):
    if n == 0 or W == 0:
        return 0

    if wt[n - 1] > W:
        return knapsack_force(W, wt, val, n - 1)
    else:
        return max(
            val[n - 1] + knapsack_force(W - wt[n - 1], wt, val, n - 1), knapsack_force(W, wt, val, n - 1))


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


def roulette_selection(genes, v, c):
    n = len(genes)

    genes_fitness = np.array([fitness_function(gene, v, c) for gene in genes])
    genes_fitness_sum = np.sum(genes_fitness)
    prob = [gene / genes_fitness_sum for gene in genes_fitness]
    print("prob", prob)
    print("prob2", np.sum(prob))
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

        '''BEST GENES'''

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
    m = 10
    # c = 850
    iterations = 10
    mutation_prob = 0.002
    V = [[values[i], weights[i]] for i in range(new_n)]

    result, last_iteration, history, top, top_one = genetic_algorithm(V[0:20],
                                                                      c, iterations,
                                                                      m, mutation_prob,
                                                                      "roulette",
                                                                      "one_point")
    result1, last_iteration1, history1, top1, top_one1 = genetic_algorithm(V[0:20],
                                                                      c, iterations,
                                                                      m, mutation_prob,
                                                                      "ranking",
                                                                      "one_point")
    result2, last_iteration2, history2, top2, top_one2 = genetic_algorithm(V[0:20],
                                                                      c, iterations,
                                                                      m, mutation_prob,
                                                                      "tournament",
                                                                      "one_point")
    result3, last_iteration3, history3, top3, top_one3 = genetic_algorithm(V[0:20],
                                                                      c, iterations,
                                                                      m, mutation_prob,
                                                                      "roulette",
                                                                      "two_point")
    result4, last_iteration4, history4, top4, top_one4 = genetic_algorithm(V[0:20],
                                                                      c, iterations,
                                                                      m, mutation_prob,
                                                                      "ranking",
                                                                      "two_point")
    result5, last_iteration5, history5, top5, top_one5 = genetic_algorithm(V[0:20],
                                                                      c, iterations,
                                                                      m, mutation_prob,
                                                                      "roulette",
                                                                      "two_point")

    print(last_iteration)
    print(history)
    print(top)

    # solution = knapsack_force(c, weights, values, new_n)
    solution = knapsack.knapsack(weights, values).solve(c)

    _, axs = plt.subplots(2, 3)
    axs[0, 0].plot(history)
    axs[0, 0].scatter([i for i in range(0, len(top))], top, color='red')
    print("best result ever ", top_one)
    print("solution: ", solution[0])

    axs[0, 1].plot(history1)
    axs[0, 1].scatter([i for i in range(0, len(top))], top1, color='red')
    print("best result ever ", top_one1)
    print("solution: ", solution[0])

    axs[0, 2].plot(history2)
    axs[0, 2].scatter([i for i in range(0, len(top))], top2, color='red')
    print("best result ever ", top_one2)
    print("solution: ", solution[0])

    axs[1, 0].plot(history3)
    axs[1, 0].scatter([i for i in range(0, len(top))], top3, color='red')
    print("best result ever ", top_one3)
    print("solution: ", solution[0])

    axs[1, 1].plot(history4)
    axs[1, 1].scatter([i for i in range(0, len(top))], top4, color='red')
    print("best result ever ", top_one4)
    print("solution: ", solution[0])

    axs[1, 2].plot(history5)
    axs[1, 2].scatter([i for i in range(0, len(top))], top5, color='red')
    print("best result ever ", top_one5)
    print("solution: ", solution[0])
    plt.show()


def task2():
    pass


'''

MAIN

'''

if __name__ == '__main__':
    task1()
