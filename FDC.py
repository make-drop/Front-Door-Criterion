# -*- coding: utf-8 -*-
"""FDC.ipynb
Computation of Front Door Criterion
"""

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Set, List
from itertools import chain, combinations

G = nx.DiGraph()
E_U = [("U_i", "X_i"), ("U_i", "Y_i"), ("e_X_i", "X_i"), ("e_M_i", "M_i"), ("e_Y_i", "Y_i")]
E_V = [("X_i", "M_i"), ("M_i", "Y_i")]
G.add_edges_from(E_U + E_V)
pos = nx.spectral_layout(G)
_ = nx.draw_networkx_labels(G, pos)
sty = {"min_source_margin": 12, "min_target_margin": 12}
_ = nx.draw_networkx_edges(G, pos, E_U, style="dashed", **sty)
_ = nx.draw_networkx_edges(G, pos, E_V, style="solid", **sty)


def sample_data(size: int = int(1e6), seed: int = 31):
    np.random.seed(seed)
    U_i = np.random.normal(0, 1, size)
    Z_i = np.random.normal(0, 1, size)
    e_X_i = np.random.normal(0, 1, size)
    e_M_i = np.random.normal(0, 1, size)
    e_Y_i = np.random.normal(0, 1, size)
    X_i = 0.5 * U_i + e_X_i
    M_i = 0.5 * X_i + e_M_i
    Y_i = 0.5 * M_i + 0.5 * U_i + e_Y_i
    return pd.DataFrame(
        {"X_i": X_i, "M_i": M_i, "Y_i": Y_i, "U_i": U_i, "Z_i": Z_i, "e_X_i": e_X_i, "e_M_i": e_M_i, "e_Y_i": e_Y_i})


data = sample_data(size=100000)
data.describe()


def linear_model(_data: pd.DataFrame, x: str, y: str, adjustment_set: Set[str]):
    formula = f"{y} ~ {x}"
    if len(adjustment_set) != 0:
        formula += "+" + "+".join(adjustment_set)
    estimator = sm.OLS.from_formula(formula, _data).fit()
    Y1 = estimator.predict(_data.assign(**{x: 1}))
    Y0 = estimator.predict(_data.assign(**{x: 0}))
    return np.mean(Y1 - Y0)


def augmented_graph(graph: nx.DiGraph, x: Set[str]):
    g_copy = graph.copy()
    for _X in x:
        for _L in {*g_copy.successors(_X)}:
            g_copy.remove_edge(_X, _L)
    return g_copy


def has_backdoor_paths(graph: nx.DiGraph, x: Set[str], y: Set[str], adjustment_set: Set[str]) -> bool:
    if x & y or x & adjustment_set or y & adjustment_set:
        raise Exception("X, Y and Z have to be disjointed.")
    graph_augmented = augmented_graph(graph, x)
    return not nx.d_separated(graph_augmented, x, y, adjustment_set)


def is_interceptor(graph: nx.DiGraph, x: str, y: str, adjustment_set: Set[str]) -> bool:
    g_copy = graph.copy()
    for _M in adjustment_set:
        g_copy.remove_node(_M)
    return not nx.has_path(g_copy, x, y)


def is_front_door_adjustment_set(graph: nx.DiGraph, x: str, y: str, adjustment_set: Set[str]) -> bool:
    # the adjustment set has to be descendant of X
    if adjustment_set & nx.ancestors(graph, x):
        return False
    # checks if the adjustment set intercepts all paths between X and Y
    if not is_interceptor(graph, x, y, adjustment_set):
        return False
    # check that from X to the adjustment set there are no backdoor paths
    if has_backdoor_paths(graph, {x}, adjustment_set, set()):
        return False
    # check that from the adjustment set to Y given X there are no backdoor paths
    return not has_backdoor_paths(graph.copy(), adjustment_set, {y}, {x})


def find_front_door_adjustment_set(graph: nx.DiGraph, x: str, y: str) -> List:
    # we consider only the power set of the descendants of x
    descendants_of_x = nx.descendants(graph, x)
    # we remove Y from the descendant of X
    descendants_of_x.remove(y)
    power_set_of_descendants_of_x = chain.from_iterable(
        combinations(descendants_of_x, r) for r in range(len(descendants_of_x) + 1))
    # we order the power set according to cardinality
    power_set_of_descendants_of_x = sorted(power_set_of_descendants_of_x, key=len)
    # we look for the smallest set that meets the front door criterion
    i = 0
    found = False
    adjustment_set = set()
    while not found and i < len(power_set_of_descendants_of_x):
        if is_front_door_adjustment_set(graph, x, y, set(power_set_of_descendants_of_x[i])):
            found = True
            adjustment_set = set(power_set_of_descendants_of_x[i])
        i += 1
    return [adjustment_set, found]


def compute_causal_effect_with_front_door_criterion(_data: pd.DataFrame, graph: nx.DiGraph, x: str, y: str):
    adjustment_set, found = find_front_door_adjustment_set(graph, x, y)
    if found:
        _effect = np.zeros(len(adjustment_set))
        i = 0
        # We compute for each variable K in the adjustment set the effect from X to Y
        for k in adjustment_set:
            _effect[i] = linear_model(_data, x=x, y=k, adjustment_set=set()) * \
                         linear_model(_data, x=k, y=y, adjustment_set={x})
            i = i + 1
        # The estimated effects for the different paths are summed up
        return np.sum(_effect)
    else:
        raise Exception("No adjustment set found.")


true_ace = 0.25

effect = compute_causal_effect_with_front_door_criterion(data, G, "X_i", "Y_i")

print(f"Estimated ACE: {effect:.3}, Real ACE: {true_ace:.3}, Relative Error: {(np.abs(((effect - true_ace) / true_ace) * 100)):.4}%")
