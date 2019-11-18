import taskers_utils as tu
import torch
import utils as u

class Node_Cls_Tasker():
	def __init__(self,args,dataset):
		self.data = dataset

		self.max_time = dataset.max_time

		self.args = args

		self.num_classes = 2

		self.feats_per_node = dataset.feats_per_node

		self.nodes_labels_times = dataset.nodes_labels_times

		self.get_node_feats = self.build_get_node_feats(args,dataset)

		self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)

		self.is_static = False


	def build_get_node_feats(self,args,dataset):
		if args.use_2_hot_node_feats:
			max_deg_out, max_deg_in = tu.get_max_degs(args,dataset,all_window = True)
			self.feats_per_node = max_deg_out + max_deg_in
			def get_node_feats(i,adj):
				return tu.get_2_hot_deg_feats(adj,
											  max_deg_out,
											  max_deg_in,
											  dataset.num_nodes)
		elif args.use_1_hot_node_feats:
			max_deg,_ = tu.get_max_degs(args,dataset)
			self.feats_per_node = max_deg
			def get_node_feats(i,adj):
				return tu.get_1_hot_deg_feats(adj,
											  max_deg,
											  dataset.num_nodes)
		else:
			def get_node_feats(i,adj):
				return dataset.nodes_feats#[i] I'm ignoring the index since the features for Elliptic are static

		return get_node_feats

	def build_prepare_node_feats(self,args,dataset):
		if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
			def prepare_node_feats(node_feats):
				return u.sparse_prepare_tensor(node_feats,
											   torch_size= [dataset.num_nodes,
											   				self.feats_per_node])
		# elif args.use_1_hot_node_feats:

		else:
			def prepare_node_feats(node_feats):
				return node_feats[0] #I'll have to check this up

		return prepare_node_feats

	def get_sample(self,idx,test):
		hist_adj_list = []
		hist_ndFeats_list = []
		hist_mask_list = []

		for i in range(idx - self.args.num_hist_steps, idx+1):
			#all edgess included from the beginning
			cur_adj = tu.get_sp_adj(edges = self.data.edges,
									time = i,
									weighted = True,
									time_window = self.args.adj_mat_time_window) #changed this to keep only a time window

			node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)

			node_feats = self.get_node_feats(i,cur_adj)

			cur_adj = tu.normalize_adj(adj = cur_adj, num_nodes = self.data.num_nodes)

			hist_adj_list.append(cur_adj)
			hist_ndFeats_list.append(node_feats)
			hist_mask_list.append(node_mask)

		label_adj = self.get_node_labels(idx)

		return {'idx': idx,
				'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp': label_adj,
				'node_mask_list': hist_mask_list}


	def get_node_labels(self,idx):
		# window_nodes = tu.get_sp_adj(edges = self.data.edges,
		# 							 time = idx,
		# 							 weighted = False,
		# 							 time_window = self.args.adj_mat_time_window)

		# window_nodes = window_nodes['idx'].unique()

		# fraud_times = self.data.nodes_labels_times[window_nodes]

		# non_fraudulent = ((fraud_times > idx) + (fraud_times == -1))>0
		# non_fraudulent = window_nodes[non_fraudulent]

		# fraudulent = (fraud_times <= idx) * (fraud_times > max(idx -  self.args.adj_mat_time_window,0))
		# fraudulent = window_nodes[fraudulent]

		# label_idx = torch.cat([non_fraudulent,fraudulent]).view(-1,1)
		# label_vals = torch.cat([torch.zeros(non_fraudulent.size(0)),
		# 					    torch.ones(fraudulent.size(0))])
		node_labels = self.nodes_labels_times
		subset = node_labels[:,2]==idx
		label_idx = node_labels[subset,0]
		label_vals = node_labels[subset,1]

		return {'idx': label_idx,
				'vals': label_vals}




class Static_Node_Cls_Tasker(Node_Cls_Tasker):
	def __init__(self,args,dataset):
		self.data = dataset

		self.args = args

		self.num_classes = 2



		self.adj_matrix = tu.get_static_sp_adj(edges = self.data.edges, weighted = False)

		if args.use_2_hot_node_feats:
			max_deg_out, max_deg_in = tu.get_max_degs_static(self.data.num_nodes,self.adj_matrix)
			self.feats_per_node = max_deg_out + max_deg_in
			#print ('feats_per_node',self.feats_per_node ,max_deg_out, max_deg_in)
			self.nodes_feats = tu.get_2_hot_deg_feats(self.adj_matrix ,
												  max_deg_out,
												  max_deg_in,
												  dataset.num_nodes)

			#print('XXXX self.nodes_feats',self.nodes_feats)
			self.nodes_feats = u.sparse_prepare_tensor(self.nodes_feats, torch_size= [self.data.num_nodes,self.feats_per_node], ignore_batch_dim = False)

		else:
			self.feats_per_node = dataset.feats_per_node
			self.nodes_feats = self.data.node_feats

		self.adj_matrix = tu.normalize_adj(adj = self.adj_matrix, num_nodes = self.data.num_nodes)
		self.is_static = True

	def get_sample(self,idx,test):
		#print ('self.adj_matrix',self.adj_matrix.size())
		idx=int(idx)
		#node_feats = self.data.node_feats_dict[idx]
		label = self.data.nodes_labels[idx]


		return {'idx': idx,
				#'node_feats': self.data.node_feats,
				#'adj': self.adj_matrix,
				'label': label
				}



if __name__ == '__main__':
	fraud_times = torch.tensor([10,5,3,6,7,-1,-1])
	idx = 6
	non_fraudulent = ((fraud_times > idx) + (fraud_times == -1))>0
	print(non_fraudulent)
	exit()
