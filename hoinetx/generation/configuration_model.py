import numpy as np
from collections import Counter
from hypergraph import Hypergraph


def configuration_model(h, n_steps=1000, verbose=True, label='edge', n_clash=1, detailed=False):
    def proposal_generator(m, detailed=False):
        '''
        Propose a transition in stub- and edge-labeled MH.
        '''

        def proposal(edge_list):
            i, j = np.random.randint(0, m, 2)
            f1, f2 = edge_list[i], edge_list[j]
            if detailed:
                while len(f1) != len(f2):
                    i, j = np.random.randint(0, m, 2)
                    f1, f2 = edge_list[i], edge_list[j]
            g1, g2 = pairwise_reshuffle(f1, f2, True)
            return (i, j, f1, f2, g1, g2)

        return (proposal)

    def pairwise_reshuffle(f1, f2):
        '''
        Randomly reshuffle the nodes of two edges while preserving their sizes.
        '''

        f = list(f1) + list(f2)
        s = set(f)

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
        return (tuple(sorted(g1)), tuple(sorted(g2)))

    def stub_edge_MH(h, n_steps=1000, verbose=True, label='edge', detailed=False, message=True, **kwargs):
        MH_rounds = 0
        MH_steps = 0
        C_new = [list(c) for c in h.C]
        m = len(C_new)

        proposal = proposal_generator(m, detailed)

        def MH_step(label='edge'):
            i, j, f1, f2, g1, g2 = proposal(C_new)
            C_new[i] = sorted(g1)
            C_new[j] = sorted(g2)

        n = 0

        while n < n_steps:
            MH_step()
            n += 1

        new_h = Hypergraph()
        new_h.C = [tuple(sorted(f)) for f in C_new]
        MH_steps += n
        MH_rounds += 1

        if message:
            print(str(n_steps) + ' steps completed.')

        return new_h

    def vertex_labeled_MH(h, n_steps=10000, sample_every=500, sample_fun=None, verbose=False, n_clash=0, message=True,
                          detailed=False, **kwargs):

        rand = np.random.rand
        randint = np.random.randint

        k = 0
        done = False
        c = Counter(h.C)

        epoch_num = 0
        n_rejected = 0

        m = sum(c.values())

        MH_rounds = 0
        MH_steps = 0
        acceptance_rate = 0

        while not done:
            # initialize epoch

            l = list(c.elements())

            add = []
            remove = []

            end_epoch = False
            num_clash = 0

            epoch_num += 1

            # within each epoch

            k_rand = 20000  # generate many random numbers at a time

            k_ = 0
            IJ = randint(0, m, k_rand)
            A = rand(k_rand)
            while True:
                if k_ >= k_rand / 2.0:
                    IJ = randint(0, m, k_rand)
                    A = rand(k_rand)
                    k_ = 0
                i, j = (IJ[k_], IJ[k_ + 1])
                k_ += 2

                f1, f2 = l[i], l[j]
                while f1 == f2:
                    i, j = (IJ[k_], IJ[k_ + 1])
                    k_ += 2
                    f1, f2 = l[i], l[j]
                if detailed:
                    while len(f1) != len(f2):
                        i, j = (IJ[k_], IJ[k_ + 1])
                        k_ += 2
                        f1, f2 = l[i], l[j]
                        while f1 == f2:
                            i, j = (IJ[k_], IJ[k_ + 1])
                            k_ += 2
                            f1, f2 = l[i], l[j]

                inter = 2 ** (-len((set(f1).intersection(set(f2)))))
                if A[k_] > inter / (c[f1] * c[f2]):
                    n_rejected += 1
                    k += 1
                else:  # if proposal was accepted
                    g1, g2 = pairwise_reshuffle(f1, f2, True)
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
        new_h.C = [tuple(sorted(f)) for f in list(c.elements())]
        MH_steps += k - n_rejected
        MH_rounds += 1
        acceptance_rate = (1.0 * (k - n_rejected)) / (k)
        return new_h

    def MH(h, n_steps, verbose, label, n_clash, detailed, **kwargs):
        '''
        Conduct Markov Chain Monte Carlo in order to approximately sample from the space of appropriately-labeled graphs.
        n_steps: number of steps to perform
        verbose: if True, print a finishing message with descriptive summaries of the algorithm run.
        label: the label space to use. Can take values in ['vertex' , 'stub', 'edge'].
        n_clash: the number of clashes permitted when updating the edge counts in vertex-labeled MH. n_clash = 0 will be exact but very slow. n_clash >= 2 may lead to performance gains at the cost of decreased accuracy.
        detailed: if True, preserve the number of edges of given dimension incident to each node
        **kwargs: additional arguments passed to sample_fun
        '''
        if (label == 'edge') or (label == 'stub'):
            return stub_edge_MH(h, n_steps=n_steps, verbose=verbose, label=label, detailed=detailed, **kwargs)
        elif label == 'vertex':
            return vertex_labeled_MH(h, n_steps=n_steps, verbose=verbose, n_clash=n_clash, detailed=detailed, **kwargs)
        else:
            print('not implemented')

    return MH(h, n_steps=n_steps, verbose=verbose, label=label, n_clash=n_clash, detailed=detailed)