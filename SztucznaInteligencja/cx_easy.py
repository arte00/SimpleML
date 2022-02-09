
def cx_helper(parent1, parent2):

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
    print(child_one)
    print(child_two)
    return child_one, child_two


if __name__ == '__main__':
    parent_one = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    parent_two = [9, 3, 7, 8, 2, 6, 5, 1, 4]
    parent_three = [6, 2, 7, 1, 5, 4, 8, 3]
    parent_four = [4, 1, 6, 7, 5, 2, 3, 8]
    cx_helper(parent_three, parent_four)
