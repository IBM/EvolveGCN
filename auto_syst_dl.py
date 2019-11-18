import utils as u
import os

import tarfile

import torch

from datetime import datetime


class Autonomous_Systems_Dataset():
	def __init__(self,args):
		args.aut_sys_args = u.Namespace(args.aut_sys_args)

		tar_file = os.path.join(args.aut_sys_args.folder, args.aut_sys_args.tar_file)  
		tar_archive = tarfile.open(tar_file, 'r:gz')

		self.edges = self.load_edges(args,tar_archive)

	def load_edges(self,args,tar_archive):
		files = tar_archive.getnames()

		cont_files2times = self.times_from_names(files)

		edges = []
		cols = u.Namespace({'source': 0,
							 'target': 1,
							 'time': 2})
		for file in files:
			data = u.load_data_from_tar(file, 
									tar_archive, 
									starting_line=4,
									sep='\t',
									type_fn = int,
									tensor_const = torch.LongTensor)

			time_col = torch.zeros(data.size(0),1,dtype=torch.long) + cont_files2times[file]

			data = torch.cat([data,time_col],dim = 1)

			data = torch.cat([data,data[:,[cols.target,
										   cols.source,
										   cols.time]]])
			
			edges.append(data)

		edges = torch.cat(edges)


		_,edges[:,[cols.source,cols.target]] = edges[:,[cols.source,cols.target]].unique(return_inverse = True)


		#use only first X time steps
		indices = edges[:,cols.time] < args.aut_sys_args.steps_accounted
		edges = edges[indices,:]
		
		#time aggregation
		edges[:,cols.time] = u.aggregate_by_time(edges[:,cols.time],args.aut_sys_args.aggr_time)

		self.num_nodes = int(edges[:,[cols.source,cols.target]].max()+1)


		ids = edges[:,cols.source] * self.num_nodes + edges[:,cols.target]
		self.num_non_existing = float(self.num_nodes**2 - ids.unique().size(0))


		self.max_time = edges[:,cols.time].max()
		self.min_time = edges[:,cols.time].min()
		
		return {'idx': edges, 'vals': torch.ones(edges.size(0))}

	def times_from_names(self,files):
		files2times = {}
		times2files = {}

		base = datetime.strptime("19800101", '%Y%m%d')
		for file in files:
			delta =  (datetime.strptime(file[2:-4], '%Y%m%d') - base).days

			files2times[file] = delta
			times2files[delta] = file


		cont_files2times = {}

		sorted_times = sorted(files2times.values())
		new_t = 0

		for t in sorted_times:
			
			file = times2files[t]

			cont_files2times[file] = new_t
			
			new_t += 1
		return cont_files2times