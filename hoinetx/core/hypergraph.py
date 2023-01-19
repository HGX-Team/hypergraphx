import numpy as np


class Hypergraph:

    def __init__(self, C=[]):
        self.C = [tuple(sorted(f)) for f in C]  # edge list

        self.nodes = list(set([v for f in self.C for v in f]))  # node list
        self.n = len(self.nodes)  # number of nodes

        # number of edges
        self.m = len(self.C)

        # node degree vector
        D = {}
        for i in self.nodes:
            D[i] = 0

        for f in self.C:
            for v in f:
                D[v] += 1

        self.D = D

        # edge dimension sequence
        K = np.array([len(f) for f in self.C])
        self.K = K

    def node_degrees(self, by_dimension=False):
        '''
        Return a np.array() of node degrees. If by_dimension, return a 2d np.array() 
        in which each entry gives the number of edges of each dimension incident upon the given node. 
        '''
        if not by_dimension:
            return (self.D)
        else:
            D = np.zeros((len(self.D), max(self.K)))
            for f in self.C:
                for v in f:
                    D[v, len(f) - 1] += 1
            return (D)

    def edge_dimensions(self):
        '''
        Return an np.array() of edge dimensions. 
        '''
        return (self.K)

    def node_dimension_matrix(self):
        '''
        Return a matrix in which the i,j entry gives the number of dimension j edges incident on node i. 
        '''
        A = np.zeros((self.n, max([len(f) for f in self.C]) + 1))
        for f in self.C:
            for v in f:
                A[v, len(f)] += 1
        return (A)

    def get_edges(self, node):
        '''
        Return a list of edges incident upon a specified node. 
        '''
        return ([f for f in self.C if node in f])