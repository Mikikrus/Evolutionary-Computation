import random
from itertools import combinations, product
from time import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from tqdm.auto import tqdm
from multiprocessing import Pool


def calculate_average(*args, **kwargs):
    return np.mean(args)


def calculate_cost(data: np.ndarray) -> np.ndarray:
    costs = cdist(data[:, 2].reshape(-1, 1), data[:, 2].reshape(-1, 1), calculate_average)
    distances = np.round(cdist(data[:, :2], data[:, :2], "euclidean"))
    distances = distances + costs
    return distances


def get_match_node(edge1: list, edge2: list) -> list:
    intersection = set(edge1).intersection(set(edge2))
    return list(intersection)


def sort_edges(edges: list) -> list:
    edges_sorted = [edges.pop(0)]
    last_edge = edges_sorted[-1]
    number_of_edges = len(edges)
    while len(edges_sorted) != number_of_edges:
        index = 0
        while len(edges) > index:
            matching_node = get_match_node(last_edge, edges[index])
            if matching_node:
                matching_node = matching_node[0]
                last_edge = last_edge if matching_node == last_edge[1] else last_edge[::-1]
                edge = edges.pop(index)
                edge = edge if last_edge[1] == edge[0] else edge[::-1]
                edges_sorted[-1] = last_edge
                edges_sorted.append(edge)
                last_edge = edges_sorted[-1]
                break
            index += 1
    edges_sorted.append(edges.pop(0))
    return edges_sorted


def choose_node_regret_weighted(distances: np.ndarray, edges: list, available_nodes: set, weight: float = 0.5) -> tuple:
    costs = []
    edges = np.array(edges)
    k = 0 if edges.shape[0] < 2 else 1
    for node_index in available_nodes:
        node_costs = distances[node_index][edges].sum(axis=1) - distances[edges[:, 0], edges[:, 1]]
        indicies = node_costs.argsort()
        regret = abs(node_costs[indicies[0]] - node_costs[indicies[k]])
        regret = weight * regret - (1 - weight) * node_costs[indicies[0]]
        costs.append([regret, indicies[0], node_index])
    costs = np.array(costs)
    _, edges_index, new_node_index = costs[costs[:, 0].argmax()]
    new_node_index = int(new_node_index)
    cost = (
        distances[int(new_node_index)][edges[int(edges_index)]].sum()
        - distances[edges[int(edges_index)][0], edges[int(edges_index)][1]]
    )
    return cost, int(new_node_index), int(edges_index)


def greedy_cycle(data, node_index: int, weight: float = 1):
    distances = calculate_cost(data)
    visited, edges = [], []
    limit = len(data) // 2
    cost = data[node_index][-1]
    all_nodes = set(range(len(data)))
    value = cost
    edges.append([node_index, node_index])
    visited.append(node_index)
    while len(visited) < limit:
        cost, new_node_index, best_new_node_index = choose_node_regret_weighted(
            distances=distances, edges=edges, available_nodes=all_nodes.difference(set(visited)), weight=weight
        )
        value += cost
        visited.append(int(new_node_index))
        parent_nodes = edges.pop(best_new_node_index)
        edges += [[parent_nodes[0], int(new_node_index)], [parent_nodes[1], int(new_node_index)]]
    edges = sort_edges(edges=edges)
    path = data[np.array(edges).flatten()]
    return value, path


def random_initial_solution(data, distances, i: int):
    ind = list(range(len(data)))
    value = 0
    limit = len(data) // 2
    cur, cost = data[ind[i]][:-1], data[ind[i]][-1]
    path = [np.append(cur, cost)]
    order = [ind[i]]
    while len(ind) > limit:
        ind = np.delete(ind, i, 0)
        i = random.randrange(len(ind))
        value += distances[order[-1]][ind[i]]
        path.append(np.append(cur, cost))
        order.append(ind[i])
        cur, cost = data[ind[i]][:-1], data[ind[i]][-1]
    value += distances[ind[i]][order[0]]
    path.append(path[0])
    return value, path, order


def create_neighborhood(order, total):
    # inter-route: one selected, one not selected
    not_selected = np.setdiff1d(list(range(total)), order)
    ind = list(range(len(order)))
    inter_route = [[0] + list(x) for x in list(product(ind, not_selected))]
    # intra-route:
    intra = list(combinations(ind, 2))
    # node exchange:
    node_ex = [[1] + list(x) for x in intra]
    # edge exchange:
    edge_ex = [[2] + list(x) for x in intra]
    return inter_route + node_ex + edge_ex


def calculate_value(data, distances, ordr):
    order = ordr.copy()
    order.append(order[0])
    value = sum([distances[x][y] for x, y in zip(order, order[1:])])
    path = [data[i] for i in order]
    return value, path


def local_search_greedy(data, distances, i: int):
    limit = len(data) // 2
    value, path, order = random_initial_solution(data, distances, i)
    change = True
    while change:
        change = False
        neighborhood = create_neighborhood(order, len(data))
        for move in random.sample(neighborhood, len(neighborhood)):
            if move[0] == 0:
                a = order[move[1] - 1]
                b = order[(move[1] + 1) % len(order)]
                old = distances[a][order[move[1]]] + distances[order[move[1]]][b]
                new = distances[a][move[2]] + distances[move[2]][b]
                if old > new:
                    change = True
                    order[move[1]] = move[2]
                    break
            elif move[0] == 1:
                a = order[(move[1] - 1) % len(order)]  # move[1]
                b = order[(move[1] + 1) % len(order)]
                c = order[(move[2] - 1) % len(order)]  # move[2]
                d = order[(move[2] + 1) % len(order)]
                old = (
                    distances[a][order[move[1]]]
                    + distances[order[move[1]]][b]
                    + distances[c][order[move[2]]]
                    + distances[order[move[2]]][d]
                )

                new_order = order.copy()
                new_order[move[1]], new_order[move[2]] = new_order[move[2]], new_order[move[1]]
                a = new_order[(move[1] - 1) % len(new_order)]  # move[1]
                b = new_order[(move[1] + 1) % len(new_order)]
                c = new_order[(move[2] - 1) % len(new_order)]  # move[2]
                d = new_order[(move[2] + 1) % len(new_order)]
                new = (
                    distances[c][order[move[1]]]
                    + distances[order[move[1]]][d]
                    + distances[a][order[move[2]]]
                    + distances[order[move[2]]][b]
                )

                if old > new:
                    change = True
                    order[move[1]], order[move[2]] = order[move[2]], order[move[1]]
                    break
            else:
                old = distances[order[move[1] - 1]][order[move[1]]] + distances[order[move[2] - 1]][order[move[2]]]
                new = distances[order[move[1] - 1]][order[move[2] - 1]] + distances[order[move[1]]][order[move[2]]]
                if old > new:
                    change = True
                    ind = sorted([move[1], move[2]])
                    if ind[0] == 0:
                        order = order[ind[1] - 1 :: -1] + order[ind[1] :]
                    else:
                        order = order[: ind[0]] + order[ind[1] - 1 : ind[0] - 1 : -1] + order[ind[1] :]
                    break
    return calculate_value(data, distances, order)


def local_search_steepest(data, distances, i: int):
    limit = len(data) // 2
    value, path, order = random_initial_solution(data, distances, i)
    best_score = 1
    while best_score:
        neighborhood = create_neighborhood(order, len(data))
        best_move = None
        best_score = 0
        for move in random.sample(neighborhood, len(neighborhood)):
            if move[0] == 0:
                a = order[move[1] - 1]
                b = order[(move[1] + 1) % len(order)]
                old = distances[a][order[move[1]]] + distances[order[move[1]]][b]
                new = distances[a][move[2]] + distances[move[2]][b]
                if old - new > best_score:
                    best_score = old - new
                    best_move = move
            elif move[0] == 1:
                a = order[(move[1] - 1) % len(order)]  # move[1]
                b = order[(move[1] + 1) % len(order)]
                c = order[(move[2] - 1) % len(order)]  # move[2]
                d = order[(move[2] + 1) % len(order)]
                old = (
                    distances[a][order[move[1]]]
                    + distances[order[move[1]]][b]
                    + distances[c][order[move[2]]]
                    + distances[order[move[2]]][d]
                )

                new_order = order.copy()
                new_order[move[1]], new_order[move[2]] = new_order[move[2]], new_order[move[1]]
                a = new_order[(move[1] - 1) % len(new_order)]  # move[1]
                b = new_order[(move[1] + 1) % len(new_order)]
                c = new_order[(move[2] - 1) % len(new_order)]  # move[2]
                d = new_order[(move[2] + 1) % len(new_order)]
                new = (
                    distances[c][order[move[1]]]
                    + distances[order[move[1]]][d]
                    + distances[a][order[move[2]]]
                    + distances[order[move[2]]][b]
                )

                if old - new > best_score:
                    best_score = old - new
                    best_move = move
            else:
                old = distances[order[move[1] - 1]][order[move[1]]] + distances[order[move[2] - 1]][order[move[2]]]
                new = distances[order[move[1] - 1]][order[move[2] - 1]] + distances[order[move[1]]][order[move[2]]]
                if old - new > best_score:
                    best_score = old - new
                    best_move = move
        if best_move is not None:
            if best_move[0] == 0:
                order[best_move[1]] = best_move[2]
            elif best_move[0] == 1:
                order[best_move[1]], order[best_move[2]] = order[best_move[2]], order[best_move[1]]
            else:
                ind = sorted([best_move[1], best_move[2]])
                if ind[0] == 0:
                    order = order[ind[1] - 1 :: -1] + order[ind[1] :]
                else:
                    order = order[: ind[0]] + order[ind[1] - 1 : ind[0] - 1 : -1] + order[ind[1] :]

    return calculate_value(data, distances, order)


def evaluate(func, data, n=200):
    distances = calculate_cost(data)
    total, worst_value, best_value, best_solution = 0, 0, float("inf"), None
    iterable = []
    for i in range(n):
        iterable.append([data, distances, i])
    pool = Pool()
    results = pool.starmap_async(func, iterable)
    start = time()
    for result in results.get():
        value, solution = result
        total += value
        if value < best_value:
            best_solution = solution
            best_value = value
        worst_value = max(worst_value, value)
    total_time = time() - start

    return dict(
        average_score=total / n,
        worst_score=worst_value,
        best_score=best_value,
        solution=np.array(best_solution).T,
        average_time=total_time / n,
    )


def summarize(func, data, title):
    evaluation_results = evaluate(func=func, data=data, n=len(data))
    (x, y, z) = evaluation_results["solution"]
    print(f"{title}:")
    print(
        f'average time: {evaluation_results["average_time"]}\tworst time: {evaluation_results["worst_time"]}\tbest time: {evaluation_results["best_time"]}'
    )

    print(
        f'average score: {evaluation_results["average_score"]}\tworst score: {evaluation_results["worst_score"]}\tbest score: {evaluation_results["best_score"]}'
    )
    print("Best solution:")

    plt.figure(figsize=(15, 10))
    plt.scatter(data[:, 0], data[:, 1], np.full(data[:, 0].size, 100), data[:, 2], cmap="gray_r")
    plt.clim(0, 2000)
    plt.colorbar().set_label("cost", labelpad=-45, y=1.025, rotation=0)
    plt.plot(x, y, "r")
    plt.xlim([-100, 4100])
    plt.ylim([-100, 2100])
    plt.title(title)
    plt.show()


def create_order(path):
    order = []
    for i in path:
        # print(path,)
        if i not in order:
            order.append(i)
    return order
def extract_path(data, solution: np.ndarray) -> np.ndarray:
    path = []
    for node in solution:
        path.append(np.where(np.all(data == node, axis=1))[0][0])
    return np.array(path)

def local_search_steepest_heuristic(data, distances, i):
    _, path = greedy_cycle(data, i, weight=0.5)
    order = create_order(extract_path(data,path))
    best_score = 1
    while best_score:
        neighborhood = create_neighborhood(order, len(data))
        best_move = None
        best_score = 0
        for move in random.sample(neighborhood, len(neighborhood)):
            if move[0] == 0:
                a = order[move[1] - 1]
                b = order[(move[1] + 1) % len(order)]
                old = distances[a][order[move[1]]] + distances[order[move[1]]][b]
                new = distances[a][move[2]] + distances[move[2]][b]
                if old - new > best_score:
                    best_score = old - new
                    best_move = move
            elif move[0] == 1:
                a = order[(move[1] - 1) % len(order)]  # move[1]
                b = order[(move[1] + 1) % len(order)]
                c = order[(move[2] - 1) % len(order)]  # move[2]
                d = order[(move[2] + 1) % len(order)]
                old = (
                    distances[a][order[move[1]]]
                    + distances[order[move[1]]][b]
                    + distances[c][order[move[2]]]
                    + distances[order[move[2]]][d]
                )

                new_order = order.copy()
                new_order[move[1]], new_order[move[2]] = new_order[move[2]], new_order[move[1]]
                a = new_order[(move[1] - 1) % len(new_order)]  # move[1]
                b = new_order[(move[1] + 1) % len(new_order)]
                c = new_order[(move[2] - 1) % len(new_order)]  # move[2]
                d = new_order[(move[2] + 1) % len(new_order)]
                new = (
                    distances[c][order[move[1]]]
                    + distances[order[move[1]]][d]
                    + distances[a][order[move[2]]]
                    + distances[order[move[2]]][b]
                )

                if old - new > best_score:
                    best_score = old - new
                    best_move = move
            else:
                old = distances[order[move[1] - 1]][order[move[1]]] + distances[order[move[2] - 1]][order[move[2]]]
                new = distances[order[move[1] - 1]][order[move[2] - 1]] + distances[order[move[1]]][order[move[2]]]
                if old - new > best_score:
                    best_score = old - new
                    best_move = move
        if best_move is not None:
            if best_move[0] == 0:
                order[best_move[1]] = best_move[2]
            elif best_move[0] == 1:
                order[best_move[1]], order[best_move[2]] = order[best_move[2]], order[best_move[1]]
            else:
                ind = sorted([best_move[1], best_move[2]])
                if ind[0] == 0:
                    order = order[ind[1] - 1 :: -1] + order[ind[1] :]
                else:
                    order = order[: ind[0]] + order[ind[1] - 1 : ind[0] - 1 : -1] + order[ind[1] :]

    return calculate_value(data, distances, order)


def local_search_greedy_heuristic(data, distances, i: int):
    _, path = greedy_cycle(data, i, weight=0.5)
    order = create_order(extract_path(data,path))
    change = True
    while change:
        change = False
        neighborhood = create_neighborhood(order, len(data))
        for move in random.sample(neighborhood, len(neighborhood)):
            if move[0] == 0:
                a = order[move[1] - 1]
                b = order[(move[1] + 1) % len(order)]
                old = distances[a][order[move[1]]] + distances[order[move[1]]][b]
                new = distances[a][move[2]] + distances[move[2]][b]
                if old > new:
                    change = True
                    order[move[1]] = move[2]
                    break
            elif move[0] == 1:
                a = order[(move[1] - 1) % len(order)]  # move[1]
                b = order[(move[1] + 1) % len(order)]
                c = order[(move[2] - 1) % len(order)]  # move[2]
                d = order[(move[2] + 1) % len(order)]
                old = (
                    distances[a][order[move[1]]]
                    + distances[order[move[1]]][b]
                    + distances[c][order[move[2]]]
                    + distances[order[move[2]]][d]
                )

                new_order = order.copy()
                new_order[move[1]], new_order[move[2]] = new_order[move[2]], new_order[move[1]]
                a = new_order[(move[1] - 1) % len(new_order)]  # move[1]
                b = new_order[(move[1] + 1) % len(new_order)]
                c = new_order[(move[2] - 1) % len(new_order)]  # move[2]
                d = new_order[(move[2] + 1) % len(new_order)]
                new = (
                    distances[c][order[move[1]]]
                    + distances[order[move[1]]][d]
                    + distances[a][order[move[2]]]
                    + distances[order[move[2]]][b]
                )

                if old > new:
                    change = True
                    order[move[1]], order[move[2]] = order[move[2]], order[move[1]]
                    break
            else:
                old = distances[order[move[1] - 1]][order[move[1]]] + distances[order[move[2] - 1]][order[move[2]]]
                new = distances[order[move[1] - 1]][order[move[2] - 1]] + distances[order[move[1]]][order[move[2]]]
                if old > new:
                    change = True
                    ind = sorted([move[1], move[2]])
                    if ind[0] == 0:
                        order = order[ind[1] - 1 :: -1] + order[ind[1] :]
                    else:
                        order = order[: ind[0]] + order[ind[1] - 1 : ind[0] - 1 : -1] + order[ind[1] :]
                    break
    return calculate_value(data, distances, order)
