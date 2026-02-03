from itertools import combinations
from math import prod

import logging
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist


class GroupAttractivenessModel:
    def __init__(
        self,
        n=200,
        balance=1,
        h_1_ii=1,
        h_2_iii=1,
        d=1.0,
        v=1.0,
        L=100,
        *,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ):
        """
        Run the Group Attractiveness Model described in

        Gallo L. et al., Higher-order modeling of face-to-face interactions, arXiv:2406.05026 (2024)

        Link: https://arxiv.org/abs/2406.05026

        The function on take as arguments:
        n = the number of agents
        balance = the fraction of agents with attribute 0 (others have attribute 1)
        h_1_ii = homophily matrix for groups of size two (should be a tuple of size two)
        h_2_iii = homophily matrix for groups of size three (should be a tuple of size four)
        d = scope range of the agents
        v = length of the steps when random walking
        L = size of the environment

        When balanace = 1, h_1_ii = 1 and h_2_iii = 1, the algorithm runs the standard Group Attractiveness Model without homophily.
        """
        if rng is not None and seed is not None:
            raise ValueError("Provide only one of seed= or rng=.")
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self.n = n
        self.n0 = int(self.n * balance)
        self.n1 = self.n - self.n0
        self.d = d
        self.v = v
        self.L = L

        self.a = self._rng.random(self.n)

        self.positions = np.array(
            list(
                zip(
                    self._rng.random(self.n) * self.L,
                    self._rng.random(self.n) * self.L,
                )
            )
        )

        self.active = self._rng.integers(0, 2, self.n) == 1

        self.r = self._rng.random(self.n)

        self.attribute = np.array(["0"] * self.n0 + ["1"] * self.n1)
        self._rng.shuffle(self.attribute)

        if type(h_1_ii) != tuple:
            h_00, h_11 = (
                h_1_ii,
                h_1_ii,
            )
        else:
            h_00, h_11 = h_1_ii

        self.h_1 = {"00": h_00, "01": 1 - h_00, "10": 1 - h_11, "11": h_11}

        if type(h_2_iii) != tuple:
            h_000, h_001, h_101, h_111 = h_2_iii, 0, 0, h_2_iii
        else:
            h_000, h_001, h_101, h_111 = h_2_iii

        self.h_2 = {
            "000": h_000,
            "001": h_001,
            "011": 1 - h_000 - h_001,
            "100": 1 - h_111 - h_101,
            "101": h_101,
            "111": h_111,
        }

        self.h = [
            self.h_1,
            self.h_2,
        ]  # Current implementation does not consider homophily in groups of size four or more

        self.groups = dict()
        for n in range(self.n):
            if self.active[n]:
                self.groups[n] = set([frozenset([n])])
            else:
                self.groups[n] = set()

        self.iterations = 0

        self.current_neighborhood = None
        self.current_neighboring_groups = None

        self.trajectories = set()
        self.projected_trajectories = set()
        self.tentative_trajectories = set()

        self.edges = set()

    def run(self, number_of_iterations, max_edges=None, verbose=False):
        import datetime

        logger = logging.getLogger(__name__)
        for _ in range(number_of_iterations):
            self.iteration()
            self.iterations += 1
            if verbose:
                if not _ % 100:
                    logger.info("[%d]: %s", self.iterations, datetime.datetime.now())

            if max_edges is not None:
                if len(self.edges) >= max_edges:
                    break

    def iteration(self):
        self.update_neighborhood()
        self.reset_groups()
        for i in range(self.n):
            if self.active[i]:
                isolated = len(self.current_neighborhood[i]) == 0
                max_a_j = (
                    0 if isolated else self.calculate_attractiveness_neighborhood(i)
                )
                p_i = 1 - max_a_j
                to_move = p_i > self._rng.random()
                if to_move:
                    self.move(i)
                else:
                    interacting_groups = self.filter_interacting_groups(i)
                    to_interact = len(interacting_groups) > 0
                    if to_interact:
                        self.update_tentative_social_trajectory(i, interacting_groups)
                    else:
                        self.to_isolated(i)
                if isolated:
                    self.to_inactive(i)
            else:
                self.to_active(i)

        self.update_social_trajectory()

    def reset_groups(self):
        for i in np.arange(self.n)[self.active]:
            self.groups[i] = set()

    def update_tentative_social_trajectory(self, i, interacting_groups):
        for group_neigh in interacting_groups:
            group = group_neigh.union(frozenset([i]))
            for n in group:
                self.groups[n].add(group)

    def update_social_trajectory(self):
        for n in np.arange(self.n)[self.active]:
            groups = self.groups[n].copy()
            for group1 in groups:
                for group2 in groups:
                    if (group1 != group2) and (group1.issubset(group2)):
                        self.groups[n].discard(group1)

            for group1 in self.groups[n]:
                if len(group1) > 1:
                    group_tuple = tuple(sorted(group1))
                    self.trajectories.add((self.iterations, group_tuple))
                    for link in combinations(group_tuple, 2):
                        edge = tuple(sorted(link))
                        self.projected_trajectories.add((self.iterations, edge))
                    if len(group1) == 2:
                        edge = tuple(sorted(group1))
                        self.edges.add(edge)

    def move(self, i):
        angle = self._rng.random() * 2 * np.pi
        self.positions[i] += [self.v * np.sin(angle), self.v * np.cos(angle)]
        self.positions[i] %= self.L

        self.to_isolated(i)

    def to_inactive(self, i):
        if self._rng.random() < 1 - self.r[i]:
            self.active[i] = 0
            self.groups[i] = set()
        else:
            self.to_isolated(i)

    def to_active(self, i):
        if self._rng.random() < self.r[i]:
            self.active[i] = 1
            self.to_isolated(i)

    def to_isolated(self, i):
        self.groups[i] = set([frozenset([i])])

    def update_neighborhood(self):
        distance_matrix = self.distance_in_a_periodic_box(
            self.positions[self.active], self.L
        )
        num_nodes = distance_matrix.shape[0]

        active_nodes = np.arange(self.n)[self.active]
        inactive_nodes = np.arange(self.n)[~self.active]

        neighbors = {}
        for i in range(num_nodes):
            distance_neighbors = distance_matrix[i, :]
            distance_neighbors = np.where(distance_neighbors < self.d)
            model_i = active_nodes[i]
            neighbors[model_i] = active_nodes[distance_neighbors]
            neighbors[model_i] = neighbors[model_i][neighbors[model_i] != model_i]
            neighbors[model_i] = set(neighbors[model_i])
        for i in inactive_nodes:
            neighbors[i] = set()
        self.current_neighborhood = neighbors

        self.update_neighboring_groups()

    def update_neighboring_groups(self):
        neighboring_groups = dict()

        for node, neighbors in self.current_neighborhood.items():
            if len(neighbors) == 0:
                neighboring_groups[node] = set()
            else:
                neighboring_groups_list = [
                    group for neigh in neighbors for group in self.groups[neigh]
                ]
                neighboring_groups[node] = set(neighboring_groups_list)
                neighboring_groups[node] = [
                    group.intersection(neighbors) for group in neighboring_groups[node]
                ]
                neighboring_groups[node] = set(neighboring_groups[node])

        self.current_neighboring_groups = neighboring_groups

    def calculate_attractiveness_neighborhood(self, i):
        neighboring_groups = self.current_neighboring_groups[i]

        attractiveness_list = [
            prod(self.a[list(group)]) for group in neighboring_groups
        ]

        return np.mean(attractiveness_list)

    def filter_interacting_groups(self, i):
        interacting_groups = set()

        h_ig = list()
        attr_i = self.attribute[i]
        for group in self.current_neighboring_groups[i]:
            attr_g = self.attribute[list(group)]
            ### The current implementation does not consider homophily in groups of four or more individuals
            ### Interactions in groups of four or more individuals are governed by the homophily matrix of groups of size three
            while len(attr_g) > 2:
                attr_g = attr_g[:-1]
            idx = len(attr_g) - 1
            attr_g = "".join(sorted(attr_g))
            conf = attr_i + attr_g
            h_ig.append((self.h[idx][conf], group))

        for h, group in h_ig:
            if self._rng.random() < h:
                interacting_groups.add(group)

        return interacting_groups

    def get_temporal_hyperedges(self):
        return list(self.trajectories)

    def get_temporal_projected_network(self):
        return list(self.projected_trajectories)

    def get_max_time(self):
        return self.iterations

    def get_attributes(self):
        return dict(enumerate(self.attribute))

    @staticmethod
    def distance_in_a_periodic_box(points, boundary):
        out = np.empty((2, points.shape[0] * (points.shape[0] - 1) // 2))
        for o, i in zip(out, points.T):
            pdist(i[:, None], "cityblock", out=o)
        out[out > boundary / 2] -= boundary
        return squareform(norm(out, axis=0))
