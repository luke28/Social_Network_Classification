import numpy as np
import tensorflow as tf

class Graph(object):
    def __init__(self, n):
        self.n = n
        self.m = 0
#each element is a numpy array, save edge features
        self.edge_set = []
        self.edge_table = [[]] * n
    
    def add_edge(u, v, feas):
        edge_table[u].append((v, m))
        edge_set.append((u, v, np.array(feas, dtype = np.float32)))
        m += 1

    def get_from(Id):
        return edge_set[Id][0]
    def get_to(Id):
        return edge_set[Id][1]
    def get_edge_feature(Id):
        return edge_set[Id][2]

def construct_tensorflow_graph():
    
# x_node is a node index vector of the center node neighbours
# x_edge is a edge index dictionary of the center node neighbors, the key is edge id and the value is a feature vector
