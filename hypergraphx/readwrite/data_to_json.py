"""
Useful functions to convert data into hypergraphs in convenient formats such as pickle and json.
"""

import csv

import networkx as nx
import pandas as pd

from hypergraphx import Hypergraph
from hypergraphx.readwrite import save_hypergraph


def load_high_school():
    dataset = "../../test_data/hs/High-School_data_2013.csv"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}
    class_school = {}
    for l in lines:
        t, a, b, c, d = l.split()
        if c != '2BIO3' or d != '2BIO3':
            continue
        t = int(t) - 1385982020
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]
        if a not in class_school:
            class_school[a] = c
        if b not in class_school:
            class_school[b] = d

    fopen.close()

    edges = {}
    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        hyperedges = list(nx.find_cliques(G))
        for hyperedge in hyperedges:
            hyperedge = tuple(sorted(hyperedge))
            if hyperedge in edges:
                edges[hyperedge] += 1
            else:
                edges[hyperedge] = 1

    h_graph_edges = []
    h_weight = []

    for k in edges.keys():
        h_graph_edges.append(k)
        h_weight.append(edges[k])

    H = Hypergraph(edge_list=h_graph_edges, weighted=True, weights=h_weight)
    for node in class_school:
        H.add_attr_meta(node, "class_school", class_school[node])
    save_hypergraph(H, "../../test_data/hs/hs_one_class.json", file_type="json")
    save_hypergraph(H, "../../test_data/hs/hs_one_class.pickle", file_type="pickle")
    return H

def load_primary_school():
    dataset = "../../test_data/ps/primaryschool.csv"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}
    class_school = {}
    for l in lines:
        t, a, b, c, d = l.split()
        t = int(t) - 31220
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]
        if a not in class_school:
            class_school[a] = c
        if b not in class_school:
            class_school[b] = d

    fopen.close()

    edges = {}
    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        hyperedges = list(nx.find_cliques(G))
        for hyperedge in hyperedges:
            hyperedge = tuple(sorted(hyperedge))
            if hyperedge in edges:
                edges[hyperedge] += 1
            else:
                edges[hyperedge] = 1

    h_graph_edges = []
    h_weight = []

    for k in edges.keys():
        h_graph_edges.append(k)
        h_weight.append(edges[k])

    H = Hypergraph(edge_list=h_graph_edges, weighted=True, weights=h_weight)
    for node in class_school:
        H.add_attr_meta(node, "class_school", class_school[node])
    save_hypergraph(H, "../../test_data/ps/ps.json", file_type="json")
    save_hypergraph(H, "../../test_data/ps/ps.pickle", file_type="pickle")
    return H


def load_hospital():
    import networkx as nx
    dataset = "../../test_data/hospital/hospital.dat"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}
    role = {}

    for l in lines:
        t, a, b, c, d = l.split()
        t = int(t) - 140
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

        if a not in role:
            role[a] = c
        if b not in role:
            role[b] = d

    fopen.close()

    edges = {}
    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        hyperedges = list(nx.find_cliques(G))
        for hyperedge in hyperedges:
            hyperedge = tuple(sorted(hyperedge))
            if hyperedge in edges:
                edges[hyperedge] += 1
            else:
                edges[hyperedge] = 1

    h_graph_edges = []
    h_weight = []

    for k in edges.keys():
        h_graph_edges.append(k)
        h_weight.append(edges[k])

    H = Hypergraph(edge_list=h_graph_edges, weighted=True, weights=h_weight)
    for node in role:
        H.add_attr_meta(node, "role", role[node])
    save_hypergraph(H, "../../test_data/hospital/hospital.json", file_type="json")
    save_hypergraph(H, "../../test_data/hospital/hospital.pickle", file_type="pickle")
    return edges


def load_workplace():
    import networkx as nx
    dataset = "../../test_data/workplace/workplace.dat"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}
    for l in lines:
        t, a, b = l.split()
        t = int(t) - 28820
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    edges = {}

    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        c = list(nx.find_cliques(G))
        for i in c:
            i = tuple(sorted(i))
            if i in edges:
                edges[i] += 1
            else:
                edges[i] = 1

    h_graph_edges = []
    h_weight = []

    for k in edges.keys():
        h_graph_edges.append(k)
        h_weight.append(edges[k])

    H = Hypergraph(edge_list=h_graph_edges, weighted=True, weights=h_weight)

    metadata = pd.read_csv("../../test_data/workplace/workplace_meta.csv")
    for i in range(len(metadata)):
        node = metadata["nodeName"][i]
        H.add_attr_meta(node, "class", metadata["class"][i])
        H.add_attr_meta(node, "classID", int(metadata["classID"][i]))

    save_hypergraph(H, "../../test_data/workplace/workplace.json", file_type="json")
    save_hypergraph(H, "../../test_data/workplace/workplace.pickle", file_type="pickle")
    return edges

load_workplace()


def load_gene_disease(N):
    name2id_gene = {}
    id_gene2name = {}

    diseases = {}
    idxG = 0

    tsv_file = open("DatasetHigherOrder/curated_gene_disease_associations.tsv")
    data = csv.reader(tsv_file, delimiter="\t")

    c = 0

    for row in data:
        c += 1
        if c == 1:
            continue
        gene = int(row[0])
        dis = row[4]
        if gene in name2id_gene:
            gene = name2id_gene[gene]
        else:
            name2id_gene[gene] = idxG
            id_gene2name[idxG] = gene
            gene = name2id_gene[gene]
            idxG += 1

        if dis in diseases:
            diseases[dis].append(gene)
        else:
            diseases[dis] = [gene]

    edges = set()
    tot = []

    discarded_1 = 0
    discarded = 0

    for d in diseases.keys():
        if len(diseases[d]) > 1 and len(diseases[d]) <= N:
            edges.add(tuple(sorted(diseases[d])))
        elif len(diseases[d]) == 1:
            discarded_1 += 1
        else:
            discarded += 1

        tot.append(diseases[d])

    tsv_file.close()
    return list(edges)


def pickle_PACS():
    import pandas as pd

    tb = pd.read_csv("DatasetHigherOrder/PACS.csv")

    tb = tb[['ArticleID', 'PACS', 'FullName']]

    papers = {}

    c = 0

    names = {}
    nidx = 0

    for _, row in tb.iterrows():
        idx = str(row['ArticleID'])
        a = str(row['PACS'])
        b = str(row['FullName'])

        if b in names:
            b = names[b]
        else:
            names[b] = nidx
            nidx += 1
            b = names[b]

        if idx in papers:
            papers[idx]['authors'].append(b)
        else:
            papers[idx] = {}
            papers[idx]['authors'] = [b]
            papers[idx]['PACS'] = a

        c += 1
        if c % 1000 == 0:
            print(c, tb.shape)

    import pickle
    pickle.dump(papers, open("PACS.pickle", "wb"))

    # for k in papers:
    #    print(papers[k])


def load_PACS(N):
    import pickle

    papers = pickle.load(open("PACS.pickle", "rb"))

    edges = []

    tot = []

    for k in papers:
        authors = papers[k]['authors']
        if len(authors) > 1 and len(authors) <= N:
            edges.append(tuple(sorted(authors)))
        tot.append(tuple(sorted(authors)))

    return edges


def load_PACS_single(N, S):
    import pickle

    papers = pickle.load(open("PACS.pickle", "rb"))

    edges = []

    tot = []

    for k in papers:
        if int(papers[k]['PACS']) != S:
            continue
        authors = papers[k]['authors']
        if len(authors) > 1 and len(authors) <= N:
            edges.append(tuple(sorted(authors)))
        tot.append(tuple(sorted(authors)))

    print(len(edges))
    return edges


def load_PACS_single(S):
    import pickle

    papers = pickle.load(open("PACS.pickle", "rb"))

    edges = []

    tot = []

    dist = {}
    num_per_author = {}

    for k in papers:
        if int(papers[k]['PACS']) != S:
            continue
        authors = papers[k]['authors']

        for a in authors:
            if a not in num_per_author:
                num_per_author[a] = 1
            else:
                num_per_author[a] += 1

    for k in papers:
        if int(papers[k]['PACS']) != S:
            continue
        authors = papers[k]['authors']
        if len(authors) > 1:
            check = True
            for a in authors:
                if num_per_author[a] <= 1:
                    check = False
            if check:
                edges.append(tuple(sorted(authors)))
        tot.append(tuple(sorted(authors)))

    for k in edges:
        n = len(k)
        if n not in dist:
            dist[n] = 1
        else:
            dist[n] += 1
    print(dist)

    # print(len(edges))
    return edges


def load_conference(N):
    import networkx as nx
    dataset = "DatasetHigherOrder/conference.dat"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}
    for l in lines:
        t, a, b = l.split()
        t = int(t) - 32520
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    tot = set()
    edges = set()

    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        c = list(nx.find_cliques(G))
        for i in c:
            i = tuple(sorted(i))

            if len(i) <= N:
                edges.add(i)

            tot.add(i)

    # plot_dist_hyperedges(tot, "conference")
    print(len(edges))
    return edges


def load_DBLP(N):
    dataset = "DatasetHigherOrder/dblp.csv"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}

    for i in range(len(lines)):
        if i == 0:
            continue

        l = lines[i]
        l = l.split(',')
        paper, author, y = l
        y = int(y)

        if paper in graph:
            graph[paper].append(author)
        else:
            graph[paper] = [author]

    fopen.close()

    edges = set()
    tot = set()
    for k in graph:
        p = tuple(sorted(graph[k]))
        tot.add(p)
        if len(p) > 1 and len(p) <= N:
            edges.add(p)

    # plot_dist_hyperedges(tot, "dblp")
    print(len(edges))
    return edges


def load_history(N):
    dataset = "DatasetHigherOrder/history.csv"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}

    for i in range(len(lines)):
        if i == 0:
            continue

        l = lines[i]
        l = l.split(',')
        paper, author, y = l
        y = int(y)

        if paper in graph:
            graph[paper].append(author)
        else:
            graph[paper] = [author]

    fopen.close()

    edges = set()
    tot = set()
    for k in graph:
        p = tuple(sorted(graph[k]))
        tot.add(p)
        if len(p) > 1 and len(p) <= N:
            edges.add(p)

    # plot_dist_hyperedges(tot, "history")
    print(len(edges))
    return edges


def load_geology(N):
    dataset = "DatasetHigherOrder/geology.csv"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}

    for i in range(len(lines)):
        if i == 0:
            continue

        l = lines[i]
        l = l.split(',')
        paper, author, y = l
        y = int(y)

        if paper in graph:
            graph[paper].append(author)
        else:
            graph[paper] = [author]

    fopen.close()

    edges = set()
    tot = set()
    for k in graph:
        p = tuple(sorted(graph[k]))
        tot.add(p)
        if len(p) > 1 and len(p) <= N:
            edges.add(p)

    # plot_dist_hyperedges(tot, "geology")
    print(len(edges))
    return edges


def load_justice_ideo(N):
    dataset = "DatasetHigherOrder/justice.csv"
    ideo = "DatasetHigherOrder/justices_ideology.csv"

    df = pd.read_csv(dataset)
    df = df[['caseId', 'justiceName', 'vote', 'justice']]

    I = pd.read_csv(ideo)
    I = I[['spaethid', 'ideo']]

    cases = {}
    nodes = {}
    idx = 0
    dict_ideo = {}

    for _, row in df.iterrows():
        c, _, v, n = row['caseId'], row['justiceName'], row['vote'], row['justice']

        try:
            v = int(v)  # valid vote
        except:
            continue  # not voted

        n = int(n)

        if c in cases:
            if v in cases[c]:
                cases[c][v].append(n)
            else:
                cases[c][v] = [n]
        else:
            cases[c] = {}
            cases[c][v] = [n]
    for _, row in I.iterrows():
        ID, v = row[['spaethid', 'ideo']]
        try:
            ID = int(ID)
            v = float(v)
        except:
            continue
        dict_ideo[ID] = v

    tot = set()
    edges = set()

    for k in cases:
        for v in cases[k]:
            e = tuple(sorted(cases[k][v]))
            tot.add(e)
            if len(e) > 1 and len(e) <= N:
                edges.add(e)

    # plot_dist_hyperedges(tot, "justice")
    print(len(edges))
    return edges, dict_ideo


def load_justice(N):
    dataset = "DatasetHigherOrder/justice.csv"

    df = pd.read_csv(dataset)
    df = df[['caseId', 'justiceName', 'vote']]

    cases = {}
    nodes = {}
    idx = 0

    for _, row in df.iterrows():
        c, n, v = row['caseId'], row['justiceName'], row['vote']

        try:
            v = int(v)  # valid vote
        except:
            continue  # not voted

        if n in nodes:
            n = nodes[n]
        else:
            nodes[n] = idx
            idx += 1
            n = nodes[n]

        if c in cases:
            if v in cases[c]:
                cases[c][v].append(n)
            else:
                cases[c][v] = [n]
        else:
            cases[c] = {}
            cases[c][v] = [n]

    tot = set()
    edges = set()

    for k in cases:
        for v in cases[k]:
            e = tuple(sorted(cases[k][v]))
            tot.add(e)
            if len(e) > 1 and len(e) <= N:
                edges.add(e)

    # plot_dist_hyperedges(tot, "justice")
    print(len(edges))
    return edges


def load_copenaghen(N):
    import networkx as nx
    dataset = "DatasetHigherOrder/copenaghen.csv"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}
    cont = 0
    for l in lines:
        if cont == 0:
            cont += 1
            continue

        a, b, t, _ = l.split(',')
        t = int(t)
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    tot = set()
    edges = set()

    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        c = list(nx.find_cliques(G))
        for i in c:
            i = tuple(sorted(i))

            if len(i) <= N:
                edges.add(i)

            tot.add(i)

    # plot_dist_hyperedges(tot, "copenaghen")
    print(len(edges))
    return edges


def load_haggle(N):
    import networkx as nx
    dataset = "DatasetHigherOrder/haggle.csv"

    fopen = open(dataset, 'r')
    lines = fopen.readlines()

    graph = {}
    cont = 0
    for l in lines:
        if cont == 0:
            cont += 1
            continue

        a, b, _, t = l.split(',')
        t = int(t)
        a = int(a)
        b = int(b)
        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    fopen.close()

    tot = set()
    edges = set()

    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        c = list(nx.find_cliques(G))
        for i in c:
            i = tuple(sorted(i))

            if len(i) <= N:
                edges.add(i)

            tot.add(i)

    # plot_dist_hyperedges(tot, "haggle")
    print(len(edges))
    return edges


def load_babbuini(N):
    import gzip
    import networkx as nx

    f = gzip.open("DatasetHigherOrder/babbuini.txt", 'rb')
    lines = f.readlines()

    graph = {}
    names = {}
    idx = 0

    cont = 0
    for l in lines:
        if cont == 0:
            cont = 1
            continue

        l = l.split()

        t, a, b, _, _ = l

        t = int(t)

        if a in names:
            a = names[a]
        else:
            names[a] = idx
            a = idx
            idx += 1

        if b in names:
            b = names[b]
        else:
            names[b] = idx
            b = idx
            idx += 1

        if t in graph:
            graph[t].append((a, b))
        else:
            graph[t] = [(a, b)]

    tot = set()
    edges = set()

    for k in graph.keys():
        e_k = graph[k]
        G = nx.Graph(e_k, directed=False)
        c = list(nx.find_cliques(G))
        for i in c:
            i = tuple(sorted(i))

            if len(i) <= N:
                edges.add(i)

            tot.add(i)

    print(len(edges))
    return edges


def load_NDC_substances(N):
    p = "DatasetHigherOrder/NDC-substances/"
    a = open(p + 'NDC-substances-nverts.txt')
    b = open(p + 'NDC-substances-simplices.txt')
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "NDC_substances")
    print(len(edges))
    return edges


def load_NDC_classes(N):
    p = "DatasetHigherOrder/NDC-classes/"
    a = open(p + 'NDC-classes-nverts.txt')
    b = open(p + 'NDC-classes-simplices.txt')
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "NDC_classes")
    print(len(edges))
    return edges


def load_meta_NDC_classes():
    p = "DatasetHigherOrder/NDC-classes/"
    a = open(p + 'NDC-classes-node-labels.txt')
    v = a.readlines()
    a.close()

    roles = {}

    for l in v:
        l = l.strip().split(' ')
        idx = int(l[0])
        c = l[-1]
        roles[idx] = c

    return roles


def load_senate_committees():
    p = "DatasetHigherOrder/senate-committees/"
    fopen_h = open(p + 'hyperedges-senate-committees.txt')
    fopen_n = open(p + 'node-labels-senate-committees.txt')

    h = []
    for l in fopen_h.readlines():
        l = tuple(map(int, l.strip().split(',')))
        h.append(tuple(sorted(l)))

    names = []
    for l in fopen_n.readlines():
        l = int(l)
        names.append(l)

    return h, names


def load_senate_bills():
    p = "DatasetHigherOrder/senate-bills/"
    fopen_h = open(p + 'hyperedges-senate-bills.txt')
    fopen_n = open(p + 'node-labels-senate-bills.txt')

    h = []
    for l in fopen_h.readlines():
        l = tuple(map(int, l.strip().split(',')))
        h.append(tuple(sorted(l)))

    names = []
    for l in fopen_n.readlines():
        l = int(l)
        names.append(l)

    return h, names


def load_DAWN(N):
    p = "DatasetHigherOrder/DAWN/"
    a = open(p + 'DAWN-nverts.txt')
    b = open(p + 'DAWN-simplices.txt')
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    j = 0

    for i in v:
        e = []
        LIM = j + i
        while j < LIM:
            e.append(s[j])
            j += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "DAWN")
    print(len(edges))
    return edges


def load_congress(N):
    name = "congress-bills"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open(p + '{}-nverts.txt'.format(name))
    b = open(p + '{}-simplices.txt'.format(name))
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "{}".format(name))
    print(len(edges))
    return edges


def load_eu(N):
    name = "email-Eu"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open(p + '{}-nverts.txt'.format(name))
    b = open(p + '{}-simplices.txt'.format(name))
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "{}".format(name))
    print(len(edges))
    return edges


def load_enron(N):
    name = "email-Enron"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open(p + '{}-nverts.txt'.format(name))
    b = open(p + '{}-simplices.txt'.format(name))
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "{}".format(name))
    print(len(edges))
    return edges


def load_threads_ubuntu(N):
    name = "threads-ask-ubuntu"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open(p + '{}-nverts.txt'.format(name))
    b = open(p + '{}-simplices.txt'.format(name))
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    for i in v:
        cont = 0
        e = []
        while cont < i:
            e.append(s.pop(0))
            cont += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "{}".format(name))
    print(len(edges))
    return edges


def load_threads_math(N):
    name = "threads-math-sx"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open(p + '{}-nverts.txt'.format(name))
    b = open(p + '{}-simplices.txt'.format(name))
    v = list(map(int, a.readlines()))
    s = list(map(int, b.readlines()))
    a.close()
    b.close()

    edges = set()
    tot = set()

    j = 0

    for i in v:
        e = []
        LIM = j + i
        while j < LIM:
            e.append(s[j])
            j += 1
        e = tuple(sorted(e))
        if len(e) > 1 and len(e) <= N:
            edges.add(e)
        tot.add(e)

    # plot_dist_hyperedges(tot, "DAWN")
    print(len(edges))
    return edges


def load_walmart():
    name = "walmart-trips"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open('{}hyperedges-{}.txt'.format(p, name))
    l = a.readlines()
    h = set()
    for i in l:
        tmp = i.strip().split(',')
        tmp = list(map(int, tmp))
        tmp = tuple(sorted(tmp))
        h.add(tmp)
    # print(h)
    return list(h)


def load_meta_walmart():
    name = "walmart-trips"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open('{}label-names-{}.txt'.format(p, name))
    r = a.readlines()
    a.close()
    d = {}
    for i in range(len(r)):
        d[i + 1] = r[i].strip()
    a = open('{}node-labels-{}.txt'.format(p, name))
    r = a.readlines()
    a.close()
    roles = {}
    for i in range(len(r)):
        roles[i + 1] = d[int(r[i])]
    return roles


def load_math_answers():
    name = "mathoverflow-answers"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open('{}hyperedges-{}.txt'.format(p, name))
    l = a.readlines()
    h = set()
    for i in l:
        tmp = i.strip().split(',')
        tmp = list(map(int, tmp))
        tmp = tuple(sorted(tmp))
        h.add(tmp)
    # print(h)
    return list(h)


def load_MAG():
    name = "cat-edge-MAG-10"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open('{}hyperedges.txt'.format(p, name))
    l = a.readlines()
    h = set()
    for i in l:
        tmp = i.strip().split('\t')
        tmp = list(map(int, tmp))
        tmp = tuple(sorted(tmp))
        h.add(tmp)
    return list(h)


def load_recipes():
    name = "cat-edge-Cooking"
    p = "DatasetHigherOrder/{}/".format(name)
    a = open('{}hyperedges.txt'.format(p, name))
    l = a.readlines()
    h = set()
    for i in l:
        tmp = i.strip().split('\t')
        tmp = list(map(int, tmp))
        tmp = tuple(sorted(tmp))
        h.add(tmp)
    return list(h)