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

class NodeToVec(Object):
	def __init__(self, g, param):
		self.tensor_graph = tf.Graph()
		self.embedding_size = param["embedding_size"]
		self.feature_size = g.get_edge_feature(0).size
		self.num_node = g.n
		self.num_edge = g.m
		
		#construct tensorflow graph
		with tensor_graph.as_default():
			x_nodes = tf.placeholder(tf.float32, [None, num_node * embedding_size])
			embeddings = tf.Variable(
				tf.random_uniform([num_node * embedding_size], -1.0, 1.0))
			
			embededs = tf.multiply(embeddings, x_nodes)
			
			x_edges = tf.placeholder(tf.float32, [None, num_edges * feature_size])
			w_edges = tf.Variable(
				tf.random_uniform([feature_size], -1.0, 1.0))
		
	
	
	
# x_nodes is a node index vector of the center node neighbours
# x_edges is a edge index dictionary of the center node neighbors, the key is edge id and the value is a feature vector
