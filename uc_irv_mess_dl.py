import utils as u
import os

import tarfile

import torch


class Uc_Irvine_Message_Dataset():
	def __init__(self,args):
		args.uc_irc_args = u.Namespace(args.uc_irc_args)

		tar_file = os.path.join(args.uc_irc_args.folder, args.uc_irc_args.tar_file)  
		tar_archive = tarfile.open(tar_file, 'r:bz2')

		self.edges = self.load_edges(args,tar_archive)

	def load_edges(self,args,tar_archive):
		data = u.load_data_from_tar(args.uc_irc_args.edges_file, 
									tar_archive, 
									starting_line=2,
									sep=' ')
		cols = u.Namespace({'source': 0,
							 'target': 1,
							 'weight': 2,
							 'time': 3})

		data = data.long()

		self.num_nodes = int(data[:,[cols.source,cols.target]].max())

		#first id should be 0 (they are already contiguous)
		data[:,[cols.source,cols.target]] -= 1

		#add edges in the other direction (simmetric)
		data = torch.cat([data,
						   data[:,[cols.target,
						   		   cols.source,
						   		   cols.weight,
						   		   cols.time]]],
						   dim=0)

		data[:,cols.time] = u.aggregate_by_time(data[:,cols.time],
									args.uc_irc_args.aggr_time)

		ids = data[:,cols.source] * self.num_nodes + data[:,cols.target]
		self.num_non_existing = float(self.num_nodes**2 - ids.unique().size(0))

		idx = data[:,[cols.source,
				   	  cols.target,
				   	  cols.time]]

		self.max_time = data[:,cols.time].max()
		self.min_time = data[:,cols.time].min()
		

		return {'idx': idx, 'vals': torch.ones(idx.size(0))}