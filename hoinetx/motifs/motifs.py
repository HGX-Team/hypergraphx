from utils import *


def compute_motifs(hypergraph, order=3):
    edges = hypergraph.get_edges()

    def motifs_order_3():
        full, visited = motifs_ho_full(edges, 3)
        standard = motifs_standard(edges, 3, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max(full[i][1], standard[i][1])))

        return res

    def motifs_order_4():
        full, visited = motifs_ho_full(edges, 4)
        not_full, visited = motifs_ho_not_full(edges, 4, visited)
        standard = motifs_standard(edges, 4, visited)

        res = []
        for i in range(len(full)):
            res.append((full[i][0], max([full[i][1], not_full[i][1], standard[i][1]])))

        return res

    if order == 3:
        motifs_order_3(edges)
    elif order == 4:
        motifs_order_4(edges)
    else:
        print("Exact computation of motifs of order > 4 is not available.")
