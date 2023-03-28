from hypergraphx.core import Hypergraph

H = Hypergraph()
H.add_edge([1, 2, 3, 4])
H.add_edge([1, 2, 3, 5])
H.add_edge([1, 2, 3, 6])
H.add_edge([1, 3, 7])
print(H.is_connected())