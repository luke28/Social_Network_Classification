import networkx as nx
import numpy as np
import bisect as bs
import random
import traceback

class Graph(object):
	def __init__(self, n):
		self.n = n
		self.m = 0
#each element is a numpy array, save edge features
		self.edge_set = []
		self.edge_table = [[]] * n
	
	def add_edge(self, u, v, feas):
		self.edge_table[u].append((v, m))
		self.edge_set.append((u, v, np.array(feas, dtype = np.float32)))
		self.m += 1

	def get_from(self, Id):
		return self.edge_set[Id][0]
	def get_to(self, Id):
		return self.edge_set[Id][1]
	def get_edge_feature(self, Id):
		return self.edge_set[Id][2]

class DataHandler(object):
	@staticmethod
	def load_data(data_path):
		g = nx.DiGraph.to_directed(nx.read_gml(data_path))
		return g
		
	@staticmethod
	def transfer_to_graph(g):
		ret = Graph(g.number_of_nodes())
		for u in g.nodes():
			for v in g[u]:
				ret.add_edge(u, v, [float(g[u][v]['value'])])
		return ret

	@staticmethod
	def value_to_prob(g):
		for u in g.nodes():
			value_sum = 0.0
			for v in g[u]:
				value_sum += float(g[u][v]['value'])
			for v in g[u]:
				g[u][v]['value'] = float(g[u][v]['value']) / value_sum
		return g

	@staticmethod
	def get_batch(nodes_x, edges_x, y_, batch_size = 100):
		sample_size = len(nodes_x)
		if batch_size > sample_size:
			raise Exception("batch_size can not be longer than sample_size")
		idx_list = range(sample_size)
		selected_idx = random.sample(idx_list, batch_size)
		batch_nodes_x = [nodes_x[i] for i in selected_idx]
		batch_edges_x = [edges_x[i] for i in selected_idx]
		batch_y_ = [y_[i] for i in selected_idx]
		return batch_nodes_x, batch_edges_x, batch_y_


	@staticmethod
	def sampling_all(g, sample_size = 1000, num_neighbor = 10):
		nodes_x = []
		edges_x = []
		y_ = []
		for i in range(sample_size):
			idx = random.randint(0, g.number_of_nodes() - 1)
			nodes, strength = DataHandler.sampling(idx, g, num_neighbor)
			y_.append([0.0] * g.number_of_nodes())
			y_[-1][idx] = 1.0
			edges_x.append(np.array(strength, dtype = np.float32).
				reshape(len(strength), 1))
			nodes_x.append(nodes)

			print nodes_x[-1]
			print edges_x[-1]
			print y_[-1]
		return nodes_x, edges_x, y_


	@staticmethod
	def sampling(u, g, num_neighbor = 10):
		nodes = []
		strength = []
		prob_list = [(0.0, -1)]
		for v in g[u]:
			prob_list.append((prob_list[-1][0] + g[u][v]['value'], v))
		prob_list[0]

		for i in range(num_neighbor):
			p = random.random()
			idx = 0
			for item in prob_list:
				if item[0] - p > np.finfo(float).eps:
					break
				idx += 1
			v = prob_list[idx][1]
			nodes.append(v)
			strength.append(prob_list[idx][0] - prob_list[idx-1][0])

			insert_list = []
			insert_add = prob_list[idx-1][0]
			insert_mul = prob_list[idx][0] - prob_list[idx-1][0]

			for v_next in g[v]:
				insert_add += insert_mul * g[v][v_next]['value']
				insert_list.append((insert_add, v_next))
			
			del prob_list[idx]
			prob_list[idx:idx] = insert_list

		return nodes, strength
						


