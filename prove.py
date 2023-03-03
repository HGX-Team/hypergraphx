from hoinetx.core.temporal_hypergraph import TemporalHypergraph

t = TemporalHypergraph()
t.add_edge((1, (1, 2)))
t.add_edge((1, (1, 2, 3)))
t.add_edge((4, (5, 6)))
t.add_edge((1, (1, 2, 3, 4)))

a = t.get_edges((1, 2))
print(a)

h = t.aggregate((1, 2))
print(h)
