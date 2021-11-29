# -*- coding: utf-8 -*-
"""FDC.ipynb
Computation of Front Door Criterion
"""

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Set

G = nx.DiGraph()
E_V = [("X", "M"), ("M", "Y"), ("Z", "X"), ("Z", "Y")]
G.add_edges_from(E_V)  # Add exogenous and endogenous edges to graph
pos = nx.spiral_layout(G)  # Predefine graph layout
_ = nx.draw_networkx_labels(G, pos)  # Plot node labels
sty = {"min_source_margin": 12, "min_target_margin": 12}  # Set min distance from labels
_ = nx.draw_networkx_edges(G, pos, E_V, style="solid", **sty)  # Plot solid endogenous edges


def sample_data(size: int = int(1e6), seed: int = 31):
    # Set random generator seed for results reproducibility
    np.random.seed(seed)
    U = np.random.normal(0, 1, size)
    Z = np.random.normal(0, 1, size)
    e_X = np.random.normal(0, 1, size)
    e_M = np.random.normal(0, 1, size)
    e_Y = np.random.normal(0, 1, size)
    X = 0.5 * U + e_X
    M = 0.5 * X + e_M
    Y = 0.5 * M + 0.5 * U + e_Y
    return pd.DataFrame({"X": X, "M": M, "Y": Y, "Z": Z})


data = sample_data(size=100000)
data.describe()


def ACE(data: pd.DataFrame, X: str, Y: str, Z: Set[str]):
    # Define the regresion model formula
    formula = f"{Y} ~ {X}"
    if len(Z) != 0: formula += "+" + "+".join(Z)
    # Fit Ordinary Least Square regression model
    estimator = sm.OLS.from_formula(formula, data).fit()
    # Compute potential outcomes by fixing X
    Y1 = estimator.predict(data.assign(**{X: 1}))
    Y0 = estimator.predict(data.assign(**{X: 0}))
    # Compute average causal effect
    return np.mean(Y1 - Y0)


def check_backdoor(G: nx.DiGraph, X: Set[str], Y: Set[str], M: Set[str]) -> bool:
    # If X, Y, Z are not disjoint, then X and Y are d-separated by default
    if X & Y or X & M or Y & M:
        return True
    for _X in X:
        for _L in {*G.successors(_X)}:
            G.remove_edge(_X, _L)
    return nx.d_separated(G, X, Y, M)


def is_frontdoor_adjustement_set(G: nx.DiGraph, X: str, Y: str, M: Set[str]) -> bool:
    if M & nx.ancestors(G, X):
        return False
    g_copy = G.copy()
    for _M in M:
        g_copy.remove_node(_M)
    if nx.has_path(g_copy, X, Y):
        return False
    if not (check_backdoor(G.copy(), {X}, M, set())):
        return False
    return check_backdoor(G.copy(), M, {Y}, {X})


print(is_frontdoor_adjustement_set(G, X="X", Y="Y", M={"M"}))

ace = 0.25

t_1 = ACE(data, X="X", Y="M", Z=set())
t_2 = ACE(data, X="M", Y="Y", Z={"X"})
t = t_1 * t_2

print(f"Estimated ACE: {t:.3}, Real ACE: {ace:.3}, Relative Error: {(np.abs(((t - ace) / ace) * 100)):.4}%")
