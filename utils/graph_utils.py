from collections import defaultdict
from typing import Union, Tuple

import torch
from torch.utils.data import Dataset
import numpy as np

import dgl
from dgl.sampling import sample_neighbors


def r2e(
    triplets: Union[np.ndarray, torch.Tensor], num_rels: int
) -> Tuple[np.ndarray, list, list]:
    """Function for generating triplet related to each relation.

    Args:
        triples (Union[np.ndarray, torch.Tensor]): fact triple (src id, rel id, dst id)
        num_rels (int): number of relations

    Returns:
        uniq_r, r_len, e_idx (List[np.ndarray, list, list]): array of unique relations, length of relation array, entity indexes.
    """
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r + num_rels))

    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel].add(dst)
        r_to_e[rel + num_rels].add(src)
        r_to_e[rel + num_rels].add(dst)

    r_len = []
    e_idx = []
    idx = 0

    for r in uniq_r:
        r_len.append((idx, idx + len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx



def build_sub_graph(num_nodes: int, num_rels: int, triples, device: torch.device) -> dgl.DGLGraph:
    """Build subgraph function

    Args:
        num_nodes (int): number of relation
        num_rels (int): _description_
        triples (torch.Tensor): triples tensor
        device (torch.device): GPU for training
    """
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm

    triples = triples[:, [0, 1, 2]]
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({"id": node_id, "norm": norm.view(-1, 1)})
    g.apply_edges(lambda edges: {"norm": edges.dst["norm"] * edges.src["norm"]})
    g.edata["etype"] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    g.to(device)
    g.r_to_e = torch.from_numpy(np.array(r_to_e))
    return g


def append_object(
    e1: Union[np.int64, int],
    e2: Union[np.int64, int],
    r: Union[np.int64, int],
    d: dict,
):
    """Function for appending object.

    Args:
        e1 (Union[np.int64, int]): id of head entity.
        e2 (Union[np.int64, int]): id of tail entity.
        r (Union[np.int64, int]): id of relation that connect head and tail entity.
        d (dict): triplet dicttionary.
    """
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)


def add_subject(
    e1: Union[np.int64, int],
    e2: Union[np.int64, int],
    r: Union[np.int64, int],
    d: dict,
    num_rel: Union[np.int64, int],
):
    """Function for adding subject.

    Args:
        e1 (Union[np.int64, int]): id of head entity.
        e2 (Union[np.int64, int]): id of tail entity.
        r (Union[np.int64, int]): id of relation that connect head and tail entity.
        d (dict): triplet dicttionary.
        num_rel (Union[np.int64, int]): number of relation.
    """
    if not e2 in d:
        d[e2] = {}
    if not r + num_rel in d[e2]:
        d[e2][r + num_rel] = set()
    d[e2][r + num_rel].add(e1)


def add_object(
    e1: Union[np.int64, int],
    e2: Union[np.int64, int],
    r: Union[np.int64, int],
    d: dict,
    num_rel: Union[np.int64, int],
):
    """Function for adding object.

    Args:
        e1 (Union[np.int64, int]): id of head entity.
        e2 (Union[np.int64, int]): id of tail entity.
        r (Union[np.int64, int]): id of relation that connect head and tail entity.
        d (dict): triplet dicttionary.
        num_rel (Union[np.int64, int]): number of relation.
    """
    if not e1 in d:
        d[e1] = {}
    if not r in d[e1]:
        d[e1][r] = set()
    d[e1][r].add(e2)

def loader(total_data, max_batch, start_id, no_batch=False, mode="train"):
    """_summary_

    Args:
        total_data (_type_): _description_
        max_batch (_type_): _description_
        start_id (_type_): _description_
        no_batch (bool, optional): _description_. Defaults to False.
        mode (str, optional): _description_. Defaults to "train".

    Returns:
        _type_: _description_
    """
    # e_num_time = [len(da) for da in total_data]  # 每个时刻三元组的数量
    all_data = []
    all_time = []
    for t, data in enumerate(total_data):
        if mode == "train":
            if t == 0:
                continue
        if no_batch:
            all_data.append(data)
            all_time.append(start_id + t)
        else:
            g_num = (len(data) // max_batch) + 1  # 划分的组数
            indices = np.random.permutation(len(data))
            for i in range(g_num):
                if len(indices[i * max_batch : (i + 1) * max_batch]) == 0:
                    continue
                all_data.append(data[indices[i * max_batch : (i + 1) * max_batch]])
                all_time.append(start_id + t)
    return all_data, all_time


class GraphDatasetOnline(Dataset):
    def __init__(
        self, total_data:torch.Tensor, max_batch:int=100, start_id:int=0, no_batch:bool=False, mode:str="train"
    ):
        self.data = loader(total_data, max_batch, start_id, no_batch, mode)
        self.size = len(self.data[0])

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return self.size


class CollateOnline:
    def __init__(
        self,
        num_nodes,
        num_rels,
        s_f,
        s_t,
        total_length,
        name="ICEWS14s",
        encoder="rgat",
        decoder="rgat",
        max_length=10,
        all=True,
        graph=None,
        k=2,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.g = graph
        self.k = k
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.s_f = s_f
        self.s_t = s_t
        self.max_length = max_length
        self.total_length = total_length
        self.name = name
        self.all = all

    def collate_rel(self, data, data_length=400):
        triple, t = data[0]
        if self.name in ["GDELT", "ICEWS05-15"]:
            e_dix = (self.g.edata["e_s"] < t) & (self.g.edata["e_s_"] > t - data_length)
            graph = dgl.edge_subgraph(self.g, e_dix, relabel_nodes=False)
        else:
            graph = self.g
            
        # Tuple expansion
        if triple.shape[-1] == 3:
            inverse_triple = triple[:, [2, 1, 0]]
        else:
            inverse_triple = triple[:, [2, 1, 0, 3]]
        inverse_triple[:, 1] = triple[:, 1] + self.num_rels
        triple = np.vstack([triple, inverse_triple])
        data_list = {}
        
        # Calculate what is needed for sampling
        sample_list, time_list, sample_unique, time_unique, list_length = cal_length(
            triple,
            self.s_f,
            self.s_t,
            t,
            self.total_length,
            self.max_length,
            data_length=data_length,
            name=self.name,
        )
        if time_unique[-1] == t:
            time_unique = time_unique[0:-1]
        data_list["triple"] = torch.tensor(triple)
        if self.encoder in ["rgat", "regcn"] or self.decoder in [
            "rgat",
            "rgcn",
            "rgat_r",
            "rgat_r1",
        ]:
            # First sample the third-order neighbor subgraph:
            sub_node = sample_k_neighbor(graph, sample_unique, self.k)
            sub_graph = dgl.node_subgraph(graph, sub_node)  # k阶子图
            old_n_id = sub_graph.ndata[dgl.NID]
            
            # In order to further reduce the scale, the subgraph is sampled according to time, sub_d_graph
            sub_d_eid = np.in1d(sub_graph.edata["e_t"], time_unique) | (
                np.in1d(sub_graph.edata["e_rel_o"], sample_unique)
                & np.in1d(sub_graph.edata["e_rel_h"], sample_unique)
            )
            sub_d_graph = dgl.edge_subgraph(sub_graph, torch.from_numpy(sub_d_eid))
            
        # Sample sub_d_graph for decoder
        if self.decoder in [
            "rgat",
            "rgcn",
            "rgat_r",
            "rgat_x",
            "regcn",
            "rgat1",
            "rgat_r1",
        ]:
            data_list["sub_d_graph"] = sub_d_graph
            data_list["pre_d_nid"] = old_n_id[
                sub_d_graph.ndata[dgl.NID]
            ]  # Original node number of sub_d_graph
            
        
        # Then sample sub_e_graph according to e_t for encoder
        if self.encoder in ["rgat", "rgcn", "regcn", "rgat_r1"]:
            sub_e_id = np.in1d(sub_d_graph.edata["e_t"], time_unique)
            sub_e_graph = dgl.edge_subgraph(sub_d_graph, torch.from_numpy(sub_e_id))
            data_list["sub_e_graph"] = sub_e_graph
            
            data_list["pre_e_eid"] = sub_d_graph.edata[dgl.NID][
                sub_e_graph.edata[dgl.NID]
            ]  # Labels for edges of third-order subgraphs
            
            data_list["pre_e_nid"] = old_n_id[
                sub_d_graph.ndata[dgl.NID][sub_e_graph.ndata[dgl.NID]]
            ]  # The label of the original graph node

        data_list["sample_list"] = sample_list
        data_list["time_list"] = time_list
        data_list["list_length"] = list_length
        data_list["t"] = torch.tensor([t])
        data_list["sample_unique"] = sample_unique
        data_list["time_unique"] = torch.LongTensor(time_unique)
        
        return data_list


class GraphDatasetOffline(Dataset):
    def __init__(self, root_dir, loader):
        self.root = root_dir
        self.loader = loader
        self.dir_list = self.load_data(root_dir)
        self.size = len(self.dir_list)

    @staticmethod
    def load_data(data_path):
        import os

        data_dir = []
        dir_list = os.listdir(data_path)
        dir_list.sort()
        for filename in dir_list:
            data_dir.append(os.path.join(data_path, filename))
        return data_dir

    def __getitem__(self, index):
        dir_ = self.dir_list[index]
        data = self.loader(dir_)
        return data

    def __len__(self):
        return self.size


def collateOffline(
    data,
):
    """Function for 

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    data_list = {}
    data = data[0]
    data_list["sub_e_graph"] = data[0][0]
    data_list["sub_d_graph"] = data[0][1]
    data_list["pre_e_nid"] = data[1]["pre_e_nid"]
    data_list["pre_d_nid"] = data[1]["pre_d_nid"]
    data_list["t"] = data[1]["t"]
    data_list["triple"] = data[1]["triple"]
    data_list["sample_list"] = data[1]["sample_list"]
    data_list["time_list"] = data[1]["time_list"]
    data_list["list_length"] = data[1]["list_length"]
    data_list["t"] = data[1]["t"]
    data_list["sample_unique"] = data[1]["sample_unique"]
    data_list["time_unique"] = data[1]["time_unique"]
    return data_list


def cal_length(triple, s_f, s_t, t, L=365, max_length=20, data_length=400, name=None):
    """_summary_

    Args:
        triple (_type_): _description_
        s_f (_type_): _description_
        s_t (_type_): _description_
        t (_type_): _description_
        L (int, optional): _description_. Defaults to 365.
        max_length (int, optional): _description_. Defaults to 20.
        data_length (int, optional): _description_. Defaults to 400.
        name (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if name == "ICEWS05-15" or name == "GDELT":
        if t - data_length < 0:
            s_f = s_f[:, 0 : t + 1]
            s_t = s_t[:, 0 : t + 1]
        else:
            s_f = s_f[:, t - data_length : t + 1]
            s_t = s_t[:, t - data_length : t + 1]

    entity, idx = np.unique(triple[:, 0], return_inverse=True)
    s_f = s_f[entity, 0 : t + 1]  # 0到t时刻s的索引
    s_t = s_t[entity, 0 : t + 1]  # 0到t时刻s发生交互的时间
    en_l = np.zeros((len(entity), max_length), dtype=int)  # 存实体
    t_l = L * np.ones((len(entity), max_length), dtype=int)  # 存实体发生时间
    entity_set = []
    time_set = []
    length = np.zeros(len(entity), dtype=int)
    for i in range(len(entity)):
        all_time = np.unique(s_t[i])[0:-1]
        if len(all_time) == 0:
            continue
        else:
            all_entity = np.unique(s_f[i])[0:-1]
            if len(all_entity) < max_length:
                en_l[i][0 : len(all_entity)] = all_entity
                t_l[i][0 : len(all_entity)] = all_time
                length[i] = len(all_entity)
                entity_set.append(all_entity)
                time_set.append(all_time)
            else:
                en_l[i] = all_entity[-max_length:]
                t_l[i] = all_time[-max_length:]
                length[i] = max_length
                entity_set.append(all_entity[-max_length:])
                time_set.append(all_time[-max_length:])
    if len(entity_set) == 0:
        entity_set.append([0])
        time_set.append([0])
    return (
        torch.from_numpy(en_l),
        torch.from_numpy(t_l),
        torch.from_numpy(np.unique(np.concatenate(entity_set))),
        np.unique(np.concatenate(time_set)),
        torch.from_numpy(length),
    )


def sample_k_neighbor(g, seed_nodes, k):
    """_summary_

    Args:
        g (_type_): _description_
        seed_nodes (_type_): _description_
        k (_type_): _description_

    Returns:
        _type_: _description_
    """
    temp = seed_nodes
    all_nodes = seed_nodes
    for _ in range(k):
        in_nodes = np.unique(
            torch.cat(sample_neighbors(g, temp, fanout=-1, edge_dir="in").edges())
        )
        out_nodes = np.unique(
            torch.cat(sample_neighbors(g, temp, fanout=-1, edge_dir="out").edges())
        )
        temp = np.setdiff1d(np.unique((in_nodes, out_nodes)), all_nodes)
        all_nodes = np.unique(np.concatenate((all_nodes, temp)))
    return all_nodes


def upto_k_neighbor_nodes(g, seed_nodes, k):
    """_summary_

    Args:
        g (_type_): _description_
        seed_nodes (_type_): _description_
        k (_type_): _description_

    Returns:
        _type_: _description_
    """
    for _ in range(k):
        in_nodes = list(
            torch.cat(
                dgl.sampling.sample_neighbors(
                    g, seed_nodes, fanout=-1, edge_dir="in"
                ).edges()
            ).numpy()
        )
        out_nodes = list(
            torch.cat(
                dgl.sampling.sample_neighbors(
                    g, seed_nodes, fanout=-1, edge_dir="out"
                ).edges()
            ).numpy()
        )
        new_nodes = set(in_nodes + out_nodes)
        seed_nodes = list(new_nodes | set(seed_nodes))
    return seed_nodes
