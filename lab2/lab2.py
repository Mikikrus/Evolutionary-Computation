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


def extract_path(data, solution: np.ndarray) -> np.ndarray:
    path = []
    for node in solution:
        path.append(np.where(np.all(data == node, axis=1))[0][0])
    return np.array(path)


def evaluate(func, data, n=200, weight: float = 1.0):
    total, worst_value, best_value, best_solution = 0, 0, float("inf"), None
    iterable = []
    for i in range(n):
        iterable.append([data,i,weight])
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
        path=extract_path(data, np.array(best_solution)),
        average_time=total_time / n
    )