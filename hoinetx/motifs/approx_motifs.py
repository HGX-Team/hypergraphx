from hypergraph import hypergraph
from utils import *
from loaders import *
import pickle
import random
from metrics import *


def load(name):
    with open('{}'.format(str(name)), 'rb') as handle:
        b = pickle.load(handle)
        return b


def split(edges):
    res = {}
    for e in edges:
        if len(e) in res:
            res[len(e)].append(e)
        else:
            res[len(e)] = [e]
    return res


def max_card(motif):
    m = 0
    for e in motif:
        m = max(m, len(e))
    return m


def count_max_card(motif):
    m = max_card(motif)
    res = 0
    for e in motif:
        if len(e) == m:
            res += 1
    return res


def rand_motifs_3(edges, NS, a=3):
    N = 3
    mapping, labeling = generate_motifs(N)
    NS = int(NS)

    T = {}
    graph = {}
    for e in edges:
        T[tuple(sorted(e))] = 1
        for e_i in e:
            if e_i in graph:
                graph[e_i].append(e)
            else:
                graph[e_i] = [e]

    def count_motif(nodes, avoid=False):
        nodes = tuple(sorted(tuple(nodes)))
        p_nodes = power_set(nodes)

        motif = []
        for edge in p_nodes:
            if len(edge) >= 2:
                edge = tuple(sorted(list(edge)))
                if edge in T:
                    motif.append(edge)
                    if len(edge) == 3 and avoid:
                        return

        m = {}
        idx = 1
        for i in nodes:
            m[i] = idx
            idx += 1

        labeled_motif = []
        for e in motif:
            new_e = []
            for node in e:
                new_e.append(m[node])
            new_e = tuple(sorted(new_e))
            labeled_motif.append(new_e)
        labeled_motif = tuple(sorted(labeled_motif))

        if labeled_motif in labeling:
            labeling[labeled_motif] += 1

    edges = list(edges)
    edges = split(edges)

    ns = {}
    k = 1 / (a + 1)
    ns[2] = int(k * NS)
    ns[3] = int(k * a * NS)

    sampled_edges = random.choices(edges[2], k=ns[2])

    for edge in sampled_edges:
        nodes = list(edge)
        for n in nodes:
            for e_i in graph[n]:
                tmp = list(nodes)
                tmp.extend(e_i)
                tmp = set(tmp)
                tmp = list(tmp)
                k = tuple(sorted(tmp))
                if len(tmp) == 3:
                    count_motif(tmp, True)

    sampled_edges = random.choices(edges[3], k=ns[3])
    for edge in sampled_edges:
        nodes = list(edge)
        count_motif(nodes)

    out = []

    for motif in mapping.keys():
        count = 0
        for label in mapping[motif]:
            count += labeling[label]
        if max_card(motif) == 3:
            p = int(count * (len(edges[3]) / ns[3]))
        else:
            p = int(count * (len(edges[2]) / (ns[2] * len(motif))))

        out.append((motif, p))

    out = list(sorted(out))

    D = {}
    for i in range(len(out)):
        D[i] = out[i][0]

    return out


def rand_motifs_4(edges, NS, a=3, b=2, verbose=False):
    N = 4
    mapping, labeling = generate_motifs(N)

    T = {}
    graph = {}
    for e in edges:
        if len(e) <= N:
            T[tuple(sorted(e))] = 1
            for e_i in e:
                if e_i in graph:
                    graph[e_i].append(e)
                else:
                    graph[e_i] = [e]

    def count_motif(nodes, max_c):
        nodes = tuple(sorted(tuple(nodes)))
        p_nodes = power_set(nodes)

        motif = []
        for edge in p_nodes:
            if len(edge) >= 2:
                edge = tuple(sorted(list(edge)))
                if edge in T:
                    motif.append(edge)
                    if len(edge) > max_c:
                        return

        m = {}
        idx = 1
        for i in nodes:
            m[i] = idx
            idx += 1

        labeled_motif = []
        for e in motif:
            new_e = []
            for node in e:
                new_e.append(m[node])
            new_e = tuple(sorted(new_e))
            labeled_motif.append(new_e)
        labeled_motif = tuple(sorted(labeled_motif))

        if labeled_motif in labeling:
            labeling[labeled_motif] += 1

    edges = list(edges)
    edges = split(edges)

    ns = {}
    k = 1 / (a + b + 1)
    ns[2] = int(k * NS)
    ns[3] = int(k * a * NS)
    ns[4] = int(k * b * NS)
    sampled_edges = random.choices(edges[2], k=ns[2])

    i = 0
    for edge in sampled_edges:
        i += 1
        if verbose:
            print("{} of {}".format(i, len(sampled_edges)))
        vis = {}
        nodes = list(edge)
        neigh = []
        for n in nodes:
            for e_i in graph[n]:
                neigh.append(e_i)
        for e_i in neigh:
            tmp = list(nodes)
            tmp.extend(e_i)
            tmp = list(set(tmp))
            k = tuple(sorted(tmp))
            if len(tmp) == 4 and k not in vis:
                count_motif(tmp, 2)
                vis[k] = 1
            else:
                neigh2 = []
                for n2 in tmp:
                    for e_i2 in graph[n2]:
                        neigh2.append(e_i2)
                for e_i2 in neigh2:
                    tmp2 = list(tmp)
                    tmp2.extend(e_i2)
                    tmp2 = list(set(tmp2))
                    k2 = tuple(sorted(tmp2))
                    if len(tmp2) == 4 and k2 not in vis:
                        count_motif(tmp2, 2)
                        vis[k2] = 1

    sampled_edges = random.choices(edges[3], k=ns[3])
    i = 0
    for edge in sampled_edges:
        i += 1
        if verbose:
            print("{} of {}".format(i, len(sampled_edges)))
        vis = {}
        nodes = list(edge)
        for n in nodes:
            for e_i in graph[n]:
                tmp = list(nodes)
                tmp.extend(e_i)
                tmp = list(set(tmp))
                k = tuple(sorted(tmp))
                if len(tmp) == 4 and k not in vis:
                    count_motif(tmp, 3)
                    vis[k] = 1

    sampled_edges = random.choices(edges[4], k=ns[4])
    i = 0
    for edge in sampled_edges:
        i += 1
        if verbose:
            print("{} of {}".format(i, len(sampled_edges)))
        nodes = list(edge)
        count_motif(nodes, 4)

    out = []

    for motif in mapping.keys():
        count = 0
        for label in mapping[motif]:
            count += labeling[label]

        p = int(count * (len(edges[max_card(motif)]) / (ns[max_card(motif)] * count_max_card(motif))))

        out.append((motif, p))

    out = list(sorted(out))

    return out


def rand_motifs_5(edges, NS, a=3, b=2, c=2, verbose=False):
    N = 5
    labeling = {}

    def generate_all_relabelings(n, nodes):
        res = set()
        k = nodes
        relabeling_list = list(itertools.permutations([j for j in range(1, n + 1)]))
        for relabeling in relabeling_list:
            relabeling_i = relabel(k, relabeling)
            res.add((tuple(sorted(relabeling_i))))
        return res

    # mapping, labeling = generate_motifs(N)

    T = {}
    graph = {}
    for e in edges:
        if len(e) <= N:
            T[tuple(sorted(e))] = 1
            for e_i in e:
                if e_i in graph:
                    graph[e_i].append(e)
                else:
                    graph[e_i] = [e]

    def count_motif(nodes, max_c):
        nodes = tuple(sorted(tuple(nodes)))
        p_nodes = power_set(nodes)

        motif = []
        for edge in p_nodes:
            if len(edge) >= 2:
                edge = tuple(sorted(list(edge)))
                if edge in T:
                    motif.append(edge)
                    if len(edge) > max_c:
                        return

        m = {}
        idx = 1
        for i in nodes:
            m[i] = idx
            idx += 1

        labeled_motif = []
        for e in motif:
            new_e = []
            for node in e:
                new_e.append(m[node])
            new_e = tuple(sorted(new_e))
            labeled_motif.append(new_e)
        labeled_motif = tuple(sorted(labeled_motif))

        if labeled_motif in labeling:
            labeling[labeled_motif] += 1
        else:
            labeling[labeled_motif] = 1

    edges = list(edges)
    edges = split(edges)

    ns = {}
    k = 1 / (a + b + c + 1)
    ns[2] = int(k * NS)
    ns[3] = int(k * a * NS)
    ns[4] = int(k * b * NS)
    ns[5] = int(k * c * NS)
    sampled_edges = random.choices(edges[2], k=ns[2])

    i = 0
    for edge in sampled_edges:
        i += 1
        if verbose:
            print("{} of {}".format(i, len(sampled_edges)))
        vis = {}
        nodes = list(edge)
        neigh = []
        for n in nodes:
            for e_i in graph[n]:
                neigh.append(e_i)
        for e_i in neigh:
            tmp = list(nodes)
            tmp.extend(e_i)
            tmp = list(set(tmp))
            k = tuple(sorted(tmp))
            if len(tmp) == 5 and k not in vis:
                count_motif(tmp, 2)
                vis[k] = 1
            else:
                neigh2 = []
                for n2 in tmp:
                    for e_i2 in graph[n2]:
                        neigh2.append(e_i2)
                for e_i2 in neigh2:
                    tmp2 = list(tmp)
                    tmp2.extend(e_i2)
                    tmp2 = list(set(tmp2))
                    k2 = tuple(sorted(tmp2))
                    if len(tmp2) == 5 and k2 not in vis:
                        count_motif(tmp2, 2)
                        vis[k2] = 1
                    else:
                        neigh3 = []
                        for n3 in tmp2:
                            for e_i3 in graph[n3]:
                                neigh3.append(e_i3)
                        for e_i3 in neigh3:
                            tmp3 = list(tmp2)
                            tmp3.extend(e_i3)
                            tmp3 = list(set(tmp3))
                            k3 = tuple(sorted(tmp3))
                            if len(tmp2) == 5 and k3 not in vis:
                                count_motif(tmp3, 2)
                                vis[k3] = 1

    sampled_edges = random.choices(edges[3], k=ns[3])
    i = 0
    for edge in sampled_edges:
        i += 1
        if verbose:
            print("{} of {}".format(i, len(sampled_edges)))
        vis = {}
        nodes = list(edge)
        neigh = []
        for n in nodes:
            for e_i in graph[n]:
                neigh.append(e_i)
        for e_i in neigh:
            tmp = list(nodes)
            tmp.extend(e_i)
            tmp = list(set(tmp))
            k = tuple(sorted(tmp))
            if len(tmp) == 5 and k not in vis:
                count_motif(tmp, 3)
                vis[k] = 1
            else:
                neigh2 = []
                for n2 in tmp:
                    for e_i2 in graph[n2]:
                        neigh2.append(e_i2)
                for e_i2 in neigh2:
                    tmp2 = list(tmp)
                    tmp2.extend(e_i2)
                    tmp2 = list(set(tmp2))
                    k2 = tuple(sorted(tmp2))
                    if len(tmp2) == 5 and k2 not in vis:
                        count_motif(tmp2, 3)
                        vis[k2] = 1

    sampled_edges = random.choices(edges[4], k=ns[4])
    i = 0
    for edge in sampled_edges:
        i += 1
        if verbose:
            print("{} of {}".format(i, len(sampled_edges)))
        vis = {}
        nodes = list(edge)
        for n in nodes:
            for e_i in graph[n]:
                tmp = list(nodes)
                tmp.extend(e_i)
                tmp = list(set(tmp))
                k = tuple(sorted(tmp))
                if len(tmp) == 5 and k not in vis:
                    count_motif(tmp, 4)
                    vis[k] = 1

    sampled_edges = random.choices(edges[5], k=ns[5])
    i = 0
    for edge in sampled_edges:
        i += 1
        if verbose:
            print("{} of {}".format(i, len(sampled_edges)))
        nodes = list(edge)
        count_motif(nodes, 5)

    out = {}
    vis_m = {}

    for motif in labeling.keys():
        count = 0

        if motif not in vis_m:
            all_labels = generate_all_relabelings(N, motif)
            for label in all_labels:
                vis_m[label] = True
                if label in labeling.keys():
                    count += labeling[label]

            p = int(count * (len(edges[max_card(motif)]) / (ns[max_card(motif)] * count_max_card(motif))))

            out[motif] = p

    return out


if __name__ == "__main__":
    N = 4
    S = 100
    edges = load_high_school(N)
    print("Loaded {} edges".format(len(edges)))

    if N == 3:
        output = rand_motifs_3(edges, S)
    elif N == 4:
        output = rand_motifs_4(edges, S)
    elif N == 5:
        output = rand_motifs_5(edges, S)
    else:
        print("Not implemented")
        exit(0)

    print("OBSERVED DONE")

    results = []
    STEPS = len(edges) * 10
    RUN_CONFIG_MODEL = 10
    for i in range(RUN_CONFIG_MODEL):
        print("Configuration model run: ", i + 1)
        e1 = hypergraph(edges)
        e1.MH(label='stub', n_steps=STEPS)
        if N == 3:
            m1 = rand_motifs_3(e1.C, S)
        elif N == 4:
            m1 = rand_motifs_4(e1.C, S)
        elif N == 5:
            m1 = rand_motifs_5(e1.C, S)
        results.append(m1)

    res = norm_vector(diff_sum(output, results))
    print(res)