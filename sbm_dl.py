import torch
import utils as u
import os

class sbm_dataset():
    def __init__(self,args):
        assert args.task in ['link_pred'], 'sbm only implements link_pred'
        self.ecols = u.Namespace({'FromNodeId': 0,
                                  'ToNodeId': 1,
                                  'Weight': 2,
                                  'TimeStep': 3
                                })
        args.sbm_args = u.Namespace(args.sbm_args)

        #build edge data structure
        edges = self.load_edges(args.sbm_args)
        timesteps = u.aggregate_by_time(edges[:,self.ecols.TimeStep], args.sbm_args.aggr_time)
        self.max_time = timesteps.max()
        self.min_time = timesteps.min()
        print ('TIME', self.max_time, self.min_time )
        edges[:,self.ecols.TimeStep] = timesteps

        edges[:,self.ecols.Weight] = self.cluster_negs_and_positives(edges[:,self.ecols.Weight])
        self.num_classes = edges[:,self.ecols.Weight].unique().size(0)

        self.edges = self.edges_to_sp_dict(edges)
        
        #random node features
        self.num_nodes = int(self.get_num_nodes(edges))
        self.feats_per_node = args.sbm_args.feats_per_node
        self.nodes_feats = torch.rand((self.num_nodes,self.feats_per_node))

        self.num_non_existing = self.num_nodes ** 2 - edges.size(0)

    def cluster_negs_and_positives(self,ratings):
        pos_indices = ratings >= 0
        neg_indices = ratings < 0
        ratings[pos_indices] = 1
        ratings[neg_indices] = 0
        return ratings

    def prepare_node_feats(self,node_feats):
        node_feats = node_feats[0]
        return node_feats

    def edges_to_sp_dict(self,edges):
        idx = edges[:,[self.ecols.FromNodeId,
                       self.ecols.ToNodeId,
                       self.ecols.TimeStep]]

        vals = edges[:,self.ecols.Weight]
        return {'idx': idx,
                'vals': vals}

    def get_num_nodes(self,edges):
        all_ids = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        num_nodes = all_ids.max() + 1
        return num_nodes

    def load_edges(self,sbm_args, starting_line = 1):
        file = os.path.join(sbm_args.folder,sbm_args.edges_file)
        with open(file) as f:
            lines = f.read().splitlines()
        edges = [[float(r) for r in row.split(',')] for row in lines[starting_line:]]
        edges = torch.tensor(edges,dtype = torch.long)
        return edges

    def make_contigous_node_ids(self,edges):
        new_edges = edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]]
        _, new_edges = new_edges.unique(return_inverse=True)
        edges[:,[self.ecols.FromNodeId,self.ecols.ToNodeId]] = new_edges
        return edges
