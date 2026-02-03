import numpy as np


def simplicial_contagion(hypergraph, I_0, T, beta, beta_D, mu, *, seed=None, rng=None):
    """
    Simulates the contagion process on a simplicial hypergraph.
    The process is run for T time steps.
    The initial condition is given by I_0, which is a dictionary where the keys are the nodes and the values are 1 if the node is infected and 0 otherwise.
    The infection rate is beta, the three-body infection rate is beta_D, and the recovery rate is mu.
    The output is a vector of length T, where the i-th entry is the fraction of infected nodes at time i.

    Parameters
    ----------
    hypergraph : hypergraphx.Hypergraph
        The hypergraph on which the contagion process is run.

    I_0 : dictionary
        The initial condition of the contagion process.

    T : int
        The number of time steps.

    beta : float
        The infection rate.

    beta_D : float
        The three-body infection rate.

    mu : float
        The recovery rate.
    seed : int, optional (keyword-only)
        Seed for reproducibility. Ignored if `rng` is provided.
    rng : numpy.random.Generator, optional (keyword-only)
        Random number generator to use. If provided, makes the simulation reproducible without
        touching global RNG state.

    Returns
    -------
    numpy.ndarray
        The fraction of infected nodes at each time step.
    """
    if rng is not None and seed is not None:
        raise ValueError("Provide only one of seed= or rng=.")
    rng = rng if rng is not None else np.random.default_rng(seed)

    numberInf = np.linspace(0, 0, T)
    Infected = sum(I_0.values())
    numberInf[0] = Infected
    N = len(I_0)
    nodes = hypergraph.get_nodes()
    mapping = hypergraph.get_mapping()
    I_old = I_0.copy()
    t = 1

    while Infected > 0 and t < T:
        # I_new = np.copy(I_old)
        I_new = I_old.copy()

        # We run over the nodes
        for node in nodes:
            # if the node is susceptible, we run the infection process
            if I_old[node] == 0:
                # we first run the two-body infections
                neighbors = hypergraph.get_neighbors(node, order=1)
                for neigh in neighbors:
                    if I_old[neigh] == 1 and rng.random() < beta:
                        I_new[node] = 1
                        break  # if the susceptile node gets infected, we stop iterating over its neighbors
                if I_new[node] == 1:
                    continue  # if the susceptile node is already infected, we don't run the three-body processes
                # we run the three-body infections
                triplets = hypergraph.get_incident_edges(node, order=2)
                for triplet in triplets:
                    neighbors = list(triplet)
                    neighbors.remove(node)
                    neigh1, neigh2 = tuple(neighbors)
                    if (
                        I_old[neigh1] == 1
                        and I_old[neigh2] == 1
                        and rng.random() < beta_D
                    ):
                        I_new[node] = 1
                        break  # if the susceptile node gets infected, we stop iterating over the triplets
            # if the node is infected, we run the recovery process
            elif rng.random() < mu:
                I_new[node] = 0

        I_old = I_new.copy()
        Infected = sum(I_new.values())
        numberInf[t] = Infected
        t = t + 1

    return numberInf / N
