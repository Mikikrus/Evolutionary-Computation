import random
from itertools import combinations
from time import time

import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.distance import cdist
from scipy.special import softmax
from tqdm.auto import tqdm


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


def random_initial_solution(data: np.ndarray, size: int = None) -> np.ndarray:
    """
    Creates shuffled array of indices with size equal to half of the data points.
    :param data: Data points
    :type data: np.ndarray
    :return: Data indices of nodes contained in initial solutions
    :rtype: np.ndarray
    """
    size = size if size is not None else len(data) // 2
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    order = indices[:size]
    return order


def get_order_edges(order: np.ndarray) -> np.ndarray:
    order = np.append(order, order[0])
    order_edges = sliding_window_view(order.reshape(1, -1), (1, 2)).reshape(-1, 2)
    return order_edges


def calculate_value(data: np.ndarray, distances: np.ndarray, order: np.ndarray) -> tuple:
    """
    :param data: Data points
    :type data: np.ndarray
    :param distances: Distance matrix
    :type distances: np.ndarray
    :param order: Array of indices contained in the solution
    :type order: np.ndarray
    :return: Tuple containing score for the current solution and data points corresponding to the solution
    :rtype:
    """
    order = np.append(order, order[0])
    x, y = sliding_window_view(order.reshape(1, -1), (1, 2)).reshape(-1, 2).T
    value = distances[x, y].sum()
    path = data[order]
    return value, path


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


def choose_node_regret_weighted(
    distances: np.ndarray, order_edges: list, available_nodes: np.ndarray, weight: float = 0.5, k=3
) -> tuple:
    node_cost_matrix = (
        distances[available_nodes][:, order_edges].sum(axis=-1) - distances[order_edges[:, 0], order_edges[:, 1]]
    )
    partitioned_cost_matrix_indices = np.argpartition(node_cost_matrix, k, axis=1)[
        :, :k
    ]  # https://numpy.org/doc/stable/reference/generated/numpy.partition.html
    top_k_costs = np.take_along_axis(node_cost_matrix, partitioned_cost_matrix_indices, axis=1)
    partitioned_cost_matrix_indices = np.take_along_axis(
        partitioned_cost_matrix_indices, top_k_costs.argsort(axis=1), axis=1
    )
    partitioned_cost_matrix = np.take_along_axis(
        node_cost_matrix, partitioned_cost_matrix_indices, axis=1
    )  # partitioned_cost_matrix[partitioned_cost_matrix]
    regret = partitioned_cost_matrix[:, 1:].sum(axis=1)
    weighted_regret = weight * regret - (1 - weight) * partitioned_cost_matrix[:, 0]
    max_regret_index = weighted_regret.argmax()  # new_node_index
    edges_index = partitioned_cost_matrix_indices[max_regret_index, 0]
    cost = (
        distances[max_regret_index, order_edges[edges_index]].sum()
        - distances[order_edges[edges_index][0], order_edges[edges_index][1]]
    )
    return cost, available_nodes[max_regret_index], edges_index


def greedy_cycle_with_regret(
    order: np.ndarray, distances: np.ndarray, data: np.ndarray, k: int = 3, weight: float = 0.1
) -> np.ndarray:
    all_nodes = np.arange(len(data))
    order_edges = get_order_edges(order)
    value, _ = calculate_value(data, distances, order)

    while len(order) < len(data) // 2:
        cost, new_node_index, best_new_node_index = choose_node_regret_weighted(
            distances=distances,
            order_edges=order_edges,
            available_nodes=np.setdiff1d(all_nodes, order),
            weight=weight,
            k=k,
        )
        parent_nodes = order_edges[best_new_node_index]
        order_edges = np.delete(order_edges, best_new_node_index, axis=0)
        order_edges = np.insert(
            order_edges,
            [best_new_node_index, (best_new_node_index + 1) % len(order_edges)],
            np.array([[parent_nodes[0], new_node_index], [parent_nodes[1], new_node_index]]),
            0,
        )
        value += cost
        order = np.insert(order, 0, new_node_index)
    sorted_edges = np.array(sort_edges(list(order_edges)))
    final_order = sorted_edges.flatten()[::2]
    return final_order


# Delta Steepest local search


def create_node_exchange_neighborhood(order: np.ndarray, num_of_all_points: int) -> np.ndarray:
    ind = np.arange(len(order))
    # Inter-route
    indices_for_all_data = np.arange(num_of_all_points)
    not_selected = np.setdiff1d(indices_for_all_data, order)
    XX, YY = np.meshgrid(ind, not_selected)
    inter_route = np.dstack([XX, YY]).reshape(-1, 2)
    is_inter_route = np.zeros((inter_route.shape[0], 1), dtype=np.int32)
    inter_route = np.concatenate((is_inter_route, inter_route), 1)
    return inter_route


def create_neighborhood(order: np.ndarray, num_of_all_points: int) -> np.ndarray:
    """
    Creates every possible move.
    ------------
    Inter-route
        Creates every possible combination of adding a new node to the nodes contained in the solution
    Edge exchange
        Creates every possible combination of pairs contained in the solution

    :param order: Array of indices contained in the solution
    :type order: np.ndarray
    :param num_of_all_points: Number of all data points
    :type num_of_all_points: int
    :return: Neighborhood with first column defining whether move is inter-route and the following columns defining move.
    :rtype:
    """
    ind = np.arange(len(order))
    # Inter-route
    inter_route = create_node_exchange_neighborhood(order, num_of_all_points)
    # #Edge exchange
    edge_ex = np.array(list(combinations(ind, 2)))
    is_inter_route = np.ones((edge_ex.shape[0], 1), dtype=np.int32)
    edge_ex = np.concatenate((is_inter_route, edge_ex), 1)
    # return edge_ex
    neighborhood = np.vstack((inter_route, edge_ex))
    return neighborhood
    # return edge_ex


def calculate_deltas_for_node_exchange_moves(
    order: np.ndarray, distances: np.ndarray, possible_inter_moves: np.ndarray
) -> np.ndarray:

    left_neighbors = order[possible_inter_moves[:, 1] - 1]
    right_neighbors = order[np.remainder(possible_inter_moves[:, 1] + 1, len(order))]
    currently_chosen_nodes = order[possible_inter_moves[:, 1]]
    current_distances = (
        distances[left_neighbors, currently_chosen_nodes] + distances[currently_chosen_nodes, right_neighbors]
    )

    possible_new_nodes = possible_inter_moves[:, 2]
    possible_new_distances = (
        distances[left_neighbors, possible_new_nodes] + distances[possible_new_nodes, right_neighbors]
    )

    all_deltas = current_distances - possible_new_distances
    improving_moves_mask = all_deltas > 0

    improving_deltas = all_deltas[improving_moves_mask].reshape(-1, 1).astype(np.int32)

    improving_moves = possible_inter_moves[improving_moves_mask]
    improving_moves = np.concatenate((improving_deltas, improving_moves), axis=1)
    return improving_moves


def calculate_deltas_for_edge_exchange(
    order: np.ndarray, distances: np.ndarray, possible_edge_exchange_moves: np.ndarray
) -> np.ndarray:
    left_nodes = order[possible_edge_exchange_moves[:, 1]]
    left_nodes_predecessors = order[possible_edge_exchange_moves[:, 1] - 1]

    right_nodes = order[possible_edge_exchange_moves[:, 2]]
    right_nodes_predecessors = order[possible_edge_exchange_moves[:, 2] - 1]

    current_distances = (
        distances[left_nodes, left_nodes_predecessors] + distances[right_nodes, right_nodes_predecessors]
    )
    distances_after_edge_swap = (
        distances[left_nodes, right_nodes] + distances[left_nodes_predecessors, right_nodes_predecessors]
    )

    all_deltas = current_distances - distances_after_edge_swap
    improving_moves_mask = all_deltas > 0

    improving_deltas = all_deltas[improving_moves_mask].reshape(-1, 1).astype(np.int32)
    improving_moves = possible_edge_exchange_moves[improving_moves_mask]
    improving_moves = np.concatenate((improving_deltas, improving_moves), axis=1)
    return improving_moves


def calculate_deltas(order: np.ndarray, distances: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
    possible_inter_moves = neighborhood[neighborhood[:, 0] == 0]
    improving_inter_moves = calculate_deltas_for_node_exchange_moves(
        order=order, distances=distances, possible_inter_moves=possible_inter_moves
    )

    possible_edge_exchange_moves = neighborhood[neighborhood[:, 0] != 0]

    improving_edge_exchange_moves = calculate_deltas_for_edge_exchange(
        order=order, distances=distances, possible_edge_exchange_moves=possible_edge_exchange_moves
    )

    all_improving_moves = np.concatenate((improving_inter_moves, improving_edge_exchange_moves), axis=0)

    sorted_improving_moves = all_improving_moves[all_improving_moves[:, 0].argsort()[::-1]]

    return sorted_improving_moves


def remove_not_applicable_moves(deltas: np.ndarray, best_move: np.ndarray) -> np.ndarray:
    node_exchange_moves = deltas[:, 1] == 0
    edge_exchange_moves = deltas[:, 1] == 1

    cond = np.bitwise_and(node_exchange_moves, deltas[:, 3] == best_move[3])

    not_applicable_moves = cond
    return deltas[~not_applicable_moves]


def update_order_node_exchange(order: np.ndarray, best_move: np.ndarray) -> np.ndarray:
    order[best_move[2]] = best_move[3]
    return order


def update_order_edge_exchange(order: np.ndarray, best_move: np.ndarray) -> np.ndarray:
    ind = sorted([best_move[2], best_move[3]])
    if ind[0] == 0:
        order = np.concatenate((order[ind[1] - 1 :: -1], order[ind[1] :]))
    else:
        order = np.concatenate((order[: ind[0]], order[ind[1] - 1 : ind[0] - 1 : -1], order[ind[1] :]))
    return order


def get_moves_to_recalculate_node_exchange(order: np.ndarray, deltas: np.ndarray, best_move: np.ndarray) -> np.ndarray:
    node_exchange_moves = deltas[:, 1] == 0
    edge_exchange_moves = deltas[:, 1] == 1

    cond1 = np.bitwise_and(node_exchange_moves, deltas[:, 2] == best_move[2])
    cond2 = np.bitwise_and(node_exchange_moves, deltas[:, 2] == (best_move[2] - 1) % len(order))
    cond3 = np.bitwise_and(node_exchange_moves, deltas[:, 3] == best_move[3])
    cond4 = np.bitwise_and(node_exchange_moves, deltas[:, 2] == (best_move[2] + 1) % len(order))

    cond5 = np.bitwise_and(edge_exchange_moves, deltas[:, 2] == best_move[2])
    cond6 = np.bitwise_and(edge_exchange_moves, deltas[:, 2] == (best_move[2] + 1) % len(order))
    cond9 = np.bitwise_and(edge_exchange_moves, deltas[:, 3] == best_move[2])
    cond10 = np.bitwise_and(edge_exchange_moves, deltas[:, 3] == (best_move[2] + 1) % len(order))

    moves_to_remove = cond1 | cond2 | cond3 | cond4 | cond5 | cond6 | cond9 | cond10
    return moves_to_remove


def create_new_moves(order: np.ndarray, removed_node: int) -> np.ndarray:
    ind = np.arange(len(order))
    removed_node = np.array([removed_node])
    XX, YY = np.meshgrid(ind, removed_node)
    inter_route = np.dstack([XX, YY]).reshape(-1, 2)
    is_inter_route = np.zeros((inter_route.shape[0], 1), dtype=np.int32)
    inter_route = np.concatenate((is_inter_route, inter_route), 1)
    return inter_route


def get_moves_to_recalculate_edge_exchange(order:np.ndarray, deltas: np.ndarray, best_move: np.ndarray) -> np.ndarray:
    node_exchange_moves = deltas[:, 1] == 0

    cond1 = np.bitwise_and(node_exchange_moves, deltas[:, 2] == best_move[2])
    cond2 = np.bitwise_and(node_exchange_moves, deltas[:, 2] == (best_move[2] - 1) % len(order))
    cond3 = np.bitwise_and(node_exchange_moves, deltas[:, 2] == (best_move[3] - 1) % len(order))
    cond4 = np.bitwise_and(node_exchange_moves, deltas[:, 2] == best_move[3])

    moves_to_remove = cond1 | cond2 | cond3 | cond4
    return moves_to_remove


def local_search_steepest(order, distances, data):
    neighborhood = create_neighborhood(order=order, num_of_all_points=len(data))
    deltas = calculate_deltas(order, distances, neighborhood)
    counter = 0

    while len(deltas) > 0:
        counter += 1
        best_move = deltas[0]
        # print("best_move: ",best_move)
        if best_move[1] == 0:
            deltas = remove_not_applicable_moves(deltas, best_move)
            new_deltas = create_new_moves(order, order[best_move[2]])

            order = update_order_node_exchange(order, best_move=best_move)
            moves_to_recalculate = get_moves_to_recalculate_node_exchange(order, deltas, best_move)

            # print("node exchange to recalculate: \n",deltas[moves_to_recalculate])
            nodes_moves = deltas[:, 1] == 0

            deltas_to_recalculate = np.concatenate((deltas[moves_to_recalculate, 1:], new_deltas), axis=0)
            updated_deltas = calculate_deltas(order, distances, deltas_to_recalculate)
            # print("node exchange after recalculate: \n",updated_deltas)

            deltas = np.concatenate((deltas[~moves_to_recalculate], updated_deltas), axis=0)
            deltas = deltas[deltas[:, 0].argsort()[::-1]]
            # print("all deltas \n",deltas[:5])

        elif best_move[1] == 1:
            order = update_order_edge_exchange(order, best_move)
            updated_deltas = calculate_deltas(order, distances, neighborhood[neighborhood[:, 0] == 1])
            new_deltas = calculate_deltas(order, distances, create_node_exchange_neighborhood(order, len(data)))

            deltas = np.concatenate((new_deltas, updated_deltas), axis=0)
            deltas = deltas[deltas[:, 0].argsort()[::-1]]
    return order


def destroy(order, distances):
    indices = np.arange(order.shape[0])
    mask = np.ones(order.shape[0], dtype=bool)
    order_copy = np.append(order, order[0])
    x, y = sliding_window_view(order_copy.reshape(1, -1), (1, 2)).reshape(-1, 2).T
    to_delete = np.random.choice(indices, len(order) // 4, replace=False, p=softmax(distances[x, y]))
    mask[to_delete] = False
    return order[mask]


def repair(order, distances, data, k=3, weight=0.2):
    order = greedy_cycle_with_regret(order, distances, data, k, weight)
    return order
