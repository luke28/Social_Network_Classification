import numpy as np
import tensorflow as tf



class NodeToVec(object):
	def __init__(self, param):
		self.tensor_graph = tf.Graph()
		self.embedding_size = param["embedding_size"]
		self.feature_size = param["feature_size"]
		self.sampling_size = param["sampling_size"]
		self.batch_size = param["batch_size"]
		self.num_node = param["num_node"]
		self.num_edge = param["num_edge"]
		self.learnRate = param["learnRate"]
		
		#construct tensorflow graph
		with self.tensor_graph.as_default():
			# nodes indices
			self.x_nodes = tf.placeholder(tf.int32, 
				shape = [self.batch_size, self.sampling_size])
			# relevant edges features
			self.x_edges = tf.placeholder(tf.float32,
				shape = [self.batch_size, self.sampling_size, self.feature_size])
			# one hot label
			self.y_ = tf.placeholder(tf.float32,
				shape = [None, self.num_node])

			# weight and bias for edges features
			#self.w_edges = tf.Variable(
			#	tf.random_uniform([self.feature_size, 1], 0.5, 1.0))
			#self.b_edges = tf.Variable(
			#	tf.zeros([1, 1]))

			self.embeddings = tf.Variable(
				tf.random_uniform([self.num_node, 
					self.embedding_size], -1.0, 1.0))

			# weight and bias for last layer
			self.w_final = tf.Variable(
				tf.random_uniform([self.embedding_size, self.num_node], 
					-1.0, 1.0))
			self.b_final = tf.Variable(tf.zeros([self.num_node]))
			
			
			self.embed = tf.nn.embedding_lookup(self.embeddings, 
				self.x_nodes)
			#for debug
			#print("embed: ")
			#print(self.embed)

			#self.edge_reshape = tf.reshape(self.x_edges, [-1, 
			#	self.feature_size])
			#self.edge_strength = tf.sigmoid(tf.matmul(self.edge_reshape, 
			#	self.w_edges) + tf.tile(self.b_edges, 
			#		[self.batch_size * self.sampling_size, 1]))
			
			#self.edge_strength_reshape = tf.reshape(
			#	self.edge_strength, [self.batch_size, -1, 1])
			self.edge_strength_tile = tf.tile(
				self.x_edges, [1, 1, self.embedding_size])

			self.embed_wighted = tf.multiply(
				self.edge_strength_tile, self.embed)
			self.embed_sum = tf.reduce_sum(self.embed_wighted, 1)
			#self.embed_sum = tf.reshape(self.embed_wighted, [self.batch_size, -1])

			self.y = tf.nn.softmax(tf.matmul(self.embed_sum, 
				self.w_final) + self.b_final)
			self.cross_entropy = tf.reduce_mean(
				-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
		
			self.train_step = tf.train.AdamOptimizer(self.learnRate
				).minimize(self.cross_entropy)
	
	def Train(self, get_batch, epoch_num = 1001):
		with self.tensor_graph.as_default():
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				print(sess.run((self.embeddings)))
				for i in range(epoch_num):
					batch_nodes, batch_edges, batch_y = get_batch(self.batch_size)
					self.train_step.run({self.x_nodes: 
						batch_nodes, self.x_edges: batch_edges, 
						self.y_: batch_y})
					if (i % 1000 == 0):
						print(sess.run(self.embeddings))
				return sess.run(self.embeddings)


# x_nodes is a node index vector of the center node neighbours
# x_edges is a edge index dictionary of the center node neighbors, the key is edge id and the value is a feature vector
