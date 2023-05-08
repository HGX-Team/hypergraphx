from collections import Counter

import numpy as np

from hypergraphx import Hypergraph


def _cm_MCMC(hypergraph, n_steps=1000, label='edge', n_clash=1, detailed=True):
    """
    Conduct Markov Chain Monte Carlo in order to approximately
    sample from the space of appropriately-labeled graphs.
    n_steps: number of steps to perform
    label: the label space to use. Can take values in ['vertex' , 'stub', 'edge'].
    n_clash: the number of clashes permitted when updating the edge counts in vertex-labeled MH.
        n_clash = 0 will be exact but very slow.
        n_clash >= 2 may lead to performance gains at the cost of decreased accuracy.
    detailed: if True, preserve the number of edges of given dimension incident to each node
    """
    def proposal_generator(m):
        # Propose a transition in stub- and edge-labeled MH.

        def __proposal(edge_list):
            i, j = np.random.randint(0, m, 2)
            f1, f2 = edge_list[i], edge_list[j]
            if detailed:
                while len(f1) != len(f2):
                    i, j = np.random.randint(0, m, 2)
                    f1, f2 = edge_list[i], edge_list[j]
            g1, g2 = __pairwise_reshuffle(f1, f2)
            return i, j, f1, f2, g1, g2

        return __proposal

    def __pairwise_reshuffle(f1, f2):
        # Randomly reshuffle the nodes of two edges while preserving their sizes.

        f = list(f1) + list(f2)
        intersection = set(f1).intersection(set(f2))
        ix = list(intersection)
        g1 = ix.copy()
        g2 = ix.copy()

        for v in ix:
            f.remove(v)
            f.remove(v)

        for v in f:
            if (len(g1) < len(f1)) & (len(g2) < len(f2)):
                if np.random.rand() < .5:
                    g1.append(v)
                else:
                    g2.append(v)
            elif len(g1) < len(f1):
                g1.append(v)
            elif len(g2) < len(f2):
                g2.append(v)
        if len(g1) != len(f1):
            print('oops')
            print(f1, f2, g1, g2)
        return tuple(sorted(g1)), tuple(sorted(g2))

    def stub_edge_mh(message=True):
        mh_rounds = 0
        mh_steps = 0
        c_new = [list(c) for c in hypergraph.get_edges()]
        m = len(c_new)

        proposal = proposal_generator(m)

        def mh_step():
            i, j, f1, f2, g1, g2 = proposal(c_new)
            c_new[i] = sorted(g1)
            c_new[j] = sorted(g2)

        n = 0

        while n < n_steps:
            mh_step()
            n += 1

        new_h = Hypergraph()
        # check this behavior
        generated_edges = list(set([tuple(sorted(f)) for f in c_new]))
        new_h.add_edges(generated_edges)
        mh_steps += n
        mh_rounds += 1

        if message:
            print(str(n_steps) + ' steps completed.')

        return new_h

    def vertex_labeled_mh(message=True):

        rand = np.random.rand
        randint = np.random.randint

        k = 0
        done = False
        c = Counter(hypergraph._edge_list)

        epoch_num = 0
        n_rejected = 0

        m = sum(c.values())

        mh_rounds = 0
        mh_steps = 0

        while not done:
            # initialize epoch
            l = list(c.elements())

            add = []
            remove = []
            num_clash = 0
            epoch_num += 1

            # within each epoch

            k_rand = 20000  # generate many random numbers at a time

            k_ = 0
            ij = randint(0, m, k_rand)
            a = rand(k_rand)
            while True:
                if k_ >= k_rand / 2.0:
                    ij = randint(0, m, k_rand)
                    a = rand(k_rand)
                    k_ = 0
                i, j = (ij[k_], ij[k_ + 1])
                k_ += 2

                f1, f2 = l[i], l[j]
                while f1 == f2:
                    i, j = (ij[k_], ij[k_ + 1])
                    k_ += 2
                    f1, f2 = l[i], l[j]
                if detailed:
                    while len(f1) != len(f2):
                        i, j = (ij[k_], ij[k_ + 1])
                        k_ += 2
                        f1, f2 = l[i], l[j]
                        while f1 == f2:
                            i, j = (ij[k_], ij[k_ + 1])
                            k_ += 2
                            f1, f2 = l[i], l[j]

                inter = 2 ** (-len((set(f1).intersection(set(f2)))))
                if a[k_] > inter / (c[f1] * c[f2]):
                    n_rejected += 1
                    k += 1
                else:  # if proposal was accepted
                    g1, g2 = __pairwise_reshuffle(f1, f2)
                    num_clash += remove.count(f1) + remove.count(f2)
                    if (num_clash >= n_clash) & (n_clash >= 1):
                        break
                    else:
                        remove.append(f1)
                        remove.append(f2)
                        add.append(g1)
                        add.append(g2)
                        k += 1
                    if n_clash == 0:
                        break

            add = Counter(add)
            add.subtract(Counter(remove))

            c.update(add)
            done = k - n_rejected >= n_steps
        if message:
            print(str(epoch_num) + ' epochs completed, ' + str(k - n_rejected) + ' steps taken, ' + str(
                n_rejected) + ' steps rejected.')

        new_h = Hypergraph()
        new_h.add_edges([tuple(sorted(f)) for f in list(c.elements())])
        mh_steps += k - n_rejected
        mh_rounds += 1
        return new_h

    if (label == 'edge') or (label == 'stub'):
        return stub_edge_mh()
    elif label == 'vertex':
        return vertex_labeled_mh()
    else:
        print('not implemented')


def configuration_model(hypergraph, n_steps=1000, label='edge', order=None, size=None, n_clash=1, detailed=True):
    if order is not None and size is not None:
        raise ValueError('Only one of order and size can be specified.')
    if order is None and size is None:
        return _cm_MCMC(hypergraph, n_steps=n_steps, label=label, n_clash=n_clash, detailed=detailed)

    if size is None:
        size = order + 1

    tmp_h = hypergraph.get_edges(size=size, up_to=False, subhypergraph=True, keep_isolated_nodes=True)
    shuffled = _cm_MCMC(tmp_h, n_steps=n_steps, label=label, n_clash=n_clash, detailed=detailed)
    for e in hypergraph.get_edges():
        if len(e) != size:
            shuffled.add_edge(e)
    return shuffled
