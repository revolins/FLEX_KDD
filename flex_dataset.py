
import torch
from torch_geometric.data import Dataset, InMemoryDataset
import tqdm
from utils import *
from torch_geometric.utils import k_hop_subgraph
from torch_sparse import SparseTensor

class SynthDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__(dataset_name)
        self.dataset_name = dataset_name

    def get(self):
        data = torch.load(f"dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset.pt")
        return data

    def get_edge_split(self):
        split_edge = torch.load(f"dataset/{self.dataset_name}Dataset/{self.dataset_name}Dataset_split.pt")
        return split_edge

    def len(self):
        return 0

# Adapted from SEAL-OGB github: https://github.com/facebookresearch/SEAL_OGB
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

class FlexDataset(InMemoryDataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='zo', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, run=0, add_val=False, struct=False):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = int(percent) if percent >= 1.0 else percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.run = run
        self.add_val = add_val
        self.struct = struct
        super(FlexDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        if self.percent == 100:
            name = 'flex2_{}_data_run{}'.format(self.split, self.run)
        else:
            name = 'flex2_{}_data_{}_run{}'.format(self.split, self.percent, self.run)
        if self.struct and self.split == 'train':
            name += '_struct'
        name += '.pt'
        return [name]
    
    def _extract_khop_subgraph(self, src, dst, label, num_nodes):

        if self.add_val and self.split != 'train': 
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            (src, dst), self.num_hops, self.data.full_edge_index,
            relabel_nodes=False, num_nodes=num_nodes, flow='source_to_target'
        )
        else:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                (src, dst), self.num_hops, self.data.edge_index,
                relabel_nodes=False, num_nodes=num_nodes, flow='source_to_target'
            )

    
        if self.data.x is not None:
            if isinstance(self.data.x, torch.nn.Embedding): x = self.data.x(subset)
            else: x = self.data.x[subset]
        else:
            x = torch.eye(len(subset))

        z = torch.zeros(len(subset), dtype=torch.long)
        z[mapping[0]] = 1
        z[mapping[1]] = 1

        if sub_edge_index.numel() == 0: 
            sub_edge_index = torch.tensor([[mapping[0]],[mapping[1]]])
    
        sub_edge_index = to_undirected(sub_edge_index)
        data = Data(
            x=x,
            edge_index=sub_edge_index,
            y=label,
            z=z, 
            edge=torch.tensor([[mapping[0], mapping[1]]]),
            adj_t=SparseTensor(row=sub_edge_index[0], col=sub_edge_index[1]).coalesce().bool().float(),
            node_id=subset
        )

        return data

    def process(self):
        pos_edge, neg_edge = get_pos_neg_edges(self.split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes,
                                               self.struct, 
                                               self.percent)

        pos_list = []
        assert pos_edge.size(0) == 2, "Positive Edges larger than 2 in Subgraph Processing"
        assert neg_edge.size(0) == 2, "Negative Edges larger than 2 in Subgraph Processing"
        
        for (src, dst) in tqdm(pos_edge.t(), desc=f'Sampling {self.split} Positive Subgraphs'):
            pos_list.append(self._extract_khop_subgraph(src, dst, torch.ones(1), self.data.num_nodes))
            
        neg_list = []
        for (src, dst) in tqdm(neg_edge.t(), desc=f'Sampling {self.split} Negative Subgraphs'):
            neg_list.append(self._extract_khop_subgraph(src, dst, torch.zeros(1), self.data.num_nodes))

        torch.save(self.collate(pos_list + neg_list), self.processed_paths[0])
        del pos_list, neg_list


class FlexDynamicDataset(Dataset):
    def __init__(self, root, data, split_edge, num_hops, percent=100, split='train', 
                 use_coalesce=False, node_label='zo', ratio_per_hop=1.0, 
                 max_nodes_per_hop=None, add_val=False, struct=False, **kwargs):
        self.data = data
        self.split_edge = split_edge
        self.num_hops = num_hops
        self.percent = percent
        self.split = split
        self.use_coalesce = use_coalesce
        self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.directed = False
        self.add_val = add_val
        self.struct = struct
        super(FlexDynamicDataset, self).__init__(root)

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, 
                                               self.data.edge_index, 
                                               self.data.num_nodes,
                                               False, 
                                               self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 1).t().tolist()
        self.labels = [1] * pos_edge.size(1) + [0] * neg_edge.size(1)
        
    def __len__(self):
        return len(self.links)

    def len(self):
        return self.__len__()
    

    def _extract_khop_subgraph(self, src, dst, label, num_nodes):

        if self.add_val and self.split != 'train': 
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            (src, dst), self.num_hops, self.data.full_edge_index,
            relabel_nodes=True, num_nodes=num_nodes, flow='source_to_target'
        )
        else:
            subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
                (src, dst), self.num_hops, self.data.edge_index,
                relabel_nodes=True, num_nodes=num_nodes, flow='source_to_target'
            )
    
        if self.data.x is not None:
            if isinstance(self.data.x, torch.nn.Embedding): x = self.data.x(subset)
            else: x = self.data.x[subset]
        else:
            x = torch.eye(len(subset))


        z = torch.zeros(len(subset), dtype=torch.long)
        z[mapping[0]] = 1
        z[mapping[1]] = 1

        data = Data(
            x=x,
            edge_index=sub_edge_index,
            y=label,
            z=z, 
            edge=torch.tensor([[mapping[0], mapping[1]]]),
            adj_t=SparseTensor(row=sub_edge_index[0], col=sub_edge_index[1], sparse_sizes=(subset.size(0), subset.size(0))),
            node_id=subset
        )

        return data

    def get(self, idx):
        src, dst = self.links[idx]
        y = self.labels[idx]

        data = self._extract_khop_subgraph(src, dst, y, self.data.num_nodes)
       
        return data