from utils import *


def compute_motifs(hypergraph, order):
    def motifs_order_3(edges):
        N = 3
        full, visited = motifs_ho_full(edges, N)
        standard = motifs_standard(edges, N, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max(full[i][1], standard[i][1])))

        return res

    def motifs_order_4(edges):
        N = 4
        full, visited = motifs_ho_full(edges, N)
        not_full, visited = motifs_ho_not_full(edges, N, visited)
        standard = motifs_standard(edges, N, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max([full[i][1], not_full[i][1], standard[i][1]])))

        return res

    edges = hypergraph.C

    if order == 3:
        motifs_order_3(edges)
    elif order == 4:
        motifs_order_4(edges)
    else:
        print("Motifs of order > 4 not available.")
