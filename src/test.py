import networkx as nx
from data_handler import DataHandler as dh
from node_to_vec import NodeToVec



def test1():
	def get_batch(num):
		batch_nodes = [[1], [0], [2], [1]]
		batch_y = [[1.0,0.0,0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0,0.0,1.0]]
		batch_edges = [[[0.9]], [[0.9]], [[0.01]], [[0.1]]]
		return batch_nodes, batch_edges, batch_y
	param = {}
	param["embedding_size"] = 2
	param["feature_size"] = 1
	param["sampling_size"] = 1
	param["batch_size"] = 4
	param["num_node"] = 3
	param["num_edge"] = 4
	param["learnRate"] = 0.01
	nt = NodeToVec(param)
	nt.Train(get_batch)

def main():
	def get_batch(batch_size):
		return dh.get_batch(nodes_x, edges_x, y_, batch_size)
	g = nx.Graph()
	g.add_nodes_from(range(6))
	g.add_edges_from([(0, 1), (1, 2), (2, 0), (3, 4), (3, 5)], value = 1000)
	g.add_edges_from([(2, 4), (1, 3), (4, 5)], value = 1)
	g = g.to_directed()
	g = dh.value_to_prob(g)
	
	nodes_x, edges_x, y_ = dh.sampling_all(g, 10, 1)

	param = {}
	param["embedding_size"] = 5
	param["feature_size"] = 1
	param["sampling_size"] = 1
	param["batch_size"] = 10
	param["num_node"] = g.number_of_nodes()
	param["num_edge"] = g.number_of_edges()
	param["learnRate"] = 0.01
	nt = NodeToVec(param)
	embeddings = nt.Train(get_batch, 2001)

	node_list = []
	for embed in embeddings:
		dists = []
		for y in embeddings:
			dist = 0.0
			for i in range(len(embed)):
				dist += (embed[i] - y[i]) * (embed[i] - y[i])
			dists.append((dist, len(dists)))
		dists.sort()
		node_list.append(dists)

	for item in node_list:
		for it in item:
			print(str(it[0]) + "," + str(it[1]))
		print("")





if __name__ == '__main__':
	main()