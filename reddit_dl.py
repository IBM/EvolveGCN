import utils as u
import os 
from datetime import datetime
import torch

class Reddit_Dataset():
	def __init__(self,args):
		args.reddit_args = u.Namespace(args.reddit_args)
		folder = args.reddit_args.folder

		#load nodes
		cols = u.Namespace({'id': 0,
							'feats': 1})
		file = args.reddit_args.nodes_file
		file = os.path.join(folder,file)
		with open(file) as file:
			file = file.read().splitlines()
		
		ids_str_to_int = {}
		id_counter = 0

		feats = []

		for line in file:
			line = line.split(',')
			#node id
			nd_id = line[0]
			if nd_id not in ids_str_to_int.keys():
				ids_str_to_int[nd_id] = id_counter
				id_counter += 1
				nd_feats = [float(r) for r in line[1:]]
				feats.append(nd_feats)
			else:
				print('duplicate id', nd_id)
				raise Exception('duplicate_id')

		feats = torch.tensor(feats,dtype=torch.float)
		num_nodes = feats.size(0)
		
		edges = []
		not_found = 0

		#load edges in title
		edges_tmp, not_found_tmp = self.load_edges_from_file(args.reddit_args.title_edges_file,
															 folder,
															 ids_str_to_int)
		edges.extend(edges_tmp)
		not_found += not_found_tmp
		
		#load edges in bodies

		edges_tmp, not_found_tmp = self.load_edges_from_file(args.reddit_args.body_edges_file,
															 folder,
															 ids_str_to_int)
		edges.extend(edges_tmp)
		not_found += not_found_tmp

		#min time should be 0 and time aggregation
		edges = torch.LongTensor(edges)
		edges[:,2] = u.aggregate_by_time(edges[:,2],args.reddit_args.aggr_time)
		max_time = edges[:,2].max()

		#separate classes
		sp_indices = edges[:,:3].t()
		sp_values = edges[:,3]

		# sp_edges = torch.sparse.LongTensor(sp_indices
		# 									  ,sp_values,
		# 									  torch.Size([num_nodes,
		# 									  			  num_nodes,
		# 									  			  max_time+1])).coalesce()
		# vals = sp_edges._values()
		# print(vals[vals>0].sum() + vals[vals<0].sum()*-1)
		# asdf
		
		pos_mask = sp_values == 1
		neg_mask = sp_values == -1

		neg_sp_indices = sp_indices[:,neg_mask]
		neg_sp_values = sp_values[neg_mask]
		neg_sp_edges = torch.sparse.LongTensor(neg_sp_indices
											  ,neg_sp_values,
											  torch.Size([num_nodes,
											  			  num_nodes,
											  			  max_time+1])).coalesce()

		pos_sp_indices = sp_indices[:,pos_mask]
		pos_sp_values = sp_values[pos_mask]		
		
		pos_sp_edges = torch.sparse.LongTensor(pos_sp_indices
										  	  ,pos_sp_values,
											  torch.Size([num_nodes,
											  			  num_nodes,
											  			  max_time+1])).coalesce()

		#scale positive class to separate after adding
		pos_sp_edges *= 1000
		
		sp_edges = (pos_sp_edges - neg_sp_edges).coalesce()
		
		#separating negs and positive edges per edge/timestamp
		vals = sp_edges._values()
		neg_vals = vals%1000
		pos_vals = vals//1000
		#vals is simply the number of edges between two nodes at the same time_step, regardless of the edge label
		vals = pos_vals - neg_vals

		#creating labels new_vals -> the label of the edges
		new_vals = torch.zeros(vals.size(0),dtype=torch.long)
		new_vals[vals>0] = 1
		new_vals[vals<=0] = 0
		vals = pos_vals + neg_vals
		indices_labels = torch.cat([sp_edges._indices().t(),new_vals.view(-1,1)],dim=1)
		
		self.edges = {'idx': indices_labels, 'vals': vals}
		self.num_classes = 2
		self.feats_per_node = feats.size(1)
		self.num_nodes = num_nodes
		self.nodes_feats = feats
		self.max_time = max_time
		self.min_time = 0

	def prepare_node_feats(self,node_feats):
		node_feats = node_feats[0]
		return node_feats

	
	def load_edges_from_file(self,edges_file,folder,ids_str_to_int):
		edges = []
		not_found = 0

		file = edges_file
		
		file = os.path.join(folder,file)
		with open(file) as file:
			file = file.read().splitlines()

		cols = u.Namespace({'source': 0,
							'target': 1,
							'time': 3,
							'label': 4})

		base_time = datetime.strptime("19800101", '%Y%m%d')

		
		for line in file[1:]:
			fields = line.split('\t')
			sr = fields[cols.source]
			tg = fields[cols.target]

			if sr in ids_str_to_int.keys() and tg in ids_str_to_int.keys():
				sr = ids_str_to_int[sr]
				tg = ids_str_to_int[tg]

				time = fields[cols.time].split(' ')[0]
				time = datetime.strptime(time,'%Y-%m-%d')
				time = (time - base_time).days

				label = int(fields[cols.label])
				edges.append([sr,tg,time,label])
				#add the other edge to make it undirected
				edges.append([tg,sr,time,label])
			else:
				not_found+=1

		return edges, not_found


























