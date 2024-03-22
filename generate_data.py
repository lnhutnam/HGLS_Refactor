import os
import argparse
from collections import defaultdict

import dgl
from dgl import save_graphs

import numpy as np
import torch

from utils.dataset_utils import (
    load_data,
    split_by_time,
)


def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2], quad[3]] for quad in data if quad[3] == tim]
    return np.array(triples)


def generate_graph(data, time, time_num, nodes_num=None, rel_nums=None, name="Sample"):
    """Generating graph from datasets

    Args:
        data (_type_): _description_
        time (_type_): _description_
        time_num (_type_): The number of entities at each moment
        nodes_num (_type_, optional): _description_. Defaults to None.
        rel_nums (_type_, optional): _description_. Defaults to None.
        name (str, optional): _description_. Defaults to "Sample".

    Returns:
        _type_: _description_
    """
    u = []
    v = []

    rel = []
    rel_1 = []

    u_id = []  # Record the original number of the node
    t_id = []  # Record the time each node appears

    rel_r = []  # Time difference
    rel_t = []
    rel_s = []
    rel_s_ = []
    rel_h = []
    rel_o = []

    L = len(time)  # Total length of time
    time_num = np.roll(time_num, 1)
    time_num[0] = 0
    idx = np.cumsum(time_num)
    # num_t = np.zeros(L)

    for i, tim in enumerate(time):
        print(i)
        if name == "Sample":
            data_1 = get_data_with_t(data, tim)
        else:
            data_1 = data[i]
        src1, rel1, dst1 = data_1[:, [0, 1, 2]].transpose()
        uniq_v, edges = np.unique((src1, dst1), return_inverse=True)
        n_src1, n_dst1 = (
            np.reshape(edges, (2, -1)) + idx[i]
        )  # Recode nodes in the new graph
        u.append(np.concatenate((n_src1, n_dst1)))
        u_id.append(uniq_v)
        t_id.append(i * np.ones(len(uniq_v), dtype=int))
        v.append(np.concatenate((n_dst1, n_src1)))
        rel.append(np.concatenate((rel1, rel1 + rel_nums)))
        rel_1.append(np.concatenate((rel1, rel1 + rel_nums)))
        rel_r.append(0 * np.ones(len(rel1) * 2, dtype=int))
        rel_t.append(i * np.ones(len(rel1) * 2, dtype=int))
        rel_s.append(i * np.ones(len(rel1) * 2, dtype=int))
        rel_s_.append(i * np.ones(len(rel1) * 2, dtype=int))
        rel_h.append(np.concatenate((n_src1, n_dst1)))
        rel_o.append(np.concatenate((n_dst1, n_src1)))

        if i == len(time) - 1:
            break
        for j, tim in enumerate(time[i + 1 :]):
            if name == "Sample":
                data_2 = get_data_with_t(data, time[i + j + 1])
            else:
                data_2 = data[i + j + 1]
            src2, rel2, dst2 = data_2[:, [0, 1, 2]].transpose()
            uniq_v2, edges2 = np.unique((src2, dst2), return_inverse=True)
            un_entity = np.intersect1d(
                uniq_v, uniq_v2
            )  # Nodes that are repeated in the subgraph at two times
            if len(un_entity) == 0:
                print("hhh", i, tim)
                continue
            u1 = np.where(np.in1d(uniq_v, un_entity))[
                0
            ]  # Determine the index of common nodes in different subgraphs
            u2 = np.where(np.in1d(uniq_v2, un_entity))[
                0
            ]  # Determine the index of common nodes in different subgraphs
            u.append(np.concatenate((u1 + idx[i], u2 + idx[i + j + 1])))
            v.append(np.concatenate((u2 + idx[i + j + 1], u1 + idx[i])))
            rel.append(
                2 * rel_nums * np.ones(2 * len(un_entity), dtype=int)
            )  # Time type edge
            rel_1.append(
                (2 * rel_nums + j) * np.ones(2 * len(un_entity), dtype=int)
            )  # Define sizes for edges of different time types

            # rel.append((2 * rel_nums + j) * np.ones(2 * len(un_entity), dtype=int))  # Time difference
            rel_r.append(
                (j + 1) * np.ones(2 * len(un_entity), dtype=int)
            )  # Time difference
            rel_t.append(L * np.ones(2 * len(un_entity), dtype=int))
            rel_s.append(
                (i + j + 1) * np.ones(2 * len(un_entity), dtype=int)
            )  #  Head entity occurrence time
            rel_s_.append(
                i * np.ones(2 * len(un_entity), dtype=int)
            )  #  Tail entity occurrence time
            rel_h.append(np.concatenate((u1 + idx[i], u2 + idx[i + j + 1])))
            rel_o.append(np.concatenate((u2 + idx[i + j + 1], u1 + idx[i])))

    u = np.concatenate(u)
    u_id = np.concatenate(u_id)
    t_id = np.concatenate(t_id)
    v = np.concatenate(v)
    rel = np.concatenate(rel)
    rel_1 = np.concatenate(rel_1)
    rel_r = np.concatenate(rel_r)
    rel_t = np.concatenate(rel_t)
    rel_s = np.concatenate(rel_s)
    rel_s_ = np.concatenate(rel_s_)
    rel_h = np.concatenate(rel_h)
    rel_o = np.concatenate(rel_o)
    graph = dgl.graph((u, v))
    graph.edata["etype"] = torch.LongTensor(rel)  # edge relation
    graph.edata["etype1"] = torch.LongTensor(rel_1)
    graph.edata["e_r"] = torch.LongTensor(rel_r)  # Relative Time
    graph.edata["e_t"] = torch.LongTensor(rel_t)  # edge time (T is L)
    graph.edata["e_s"] = torch.LongTensor(
        rel_s
    )  # edge time (The time when the header entity occurred)
    graph.edata["e_s_"] = torch.LongTensor(
        rel_s_
    )  # edge time (The time when the tail entity occurs)
    graph.edata["e_rel_h"] = torch.LongTensor(rel_h)
    graph.edata["e_rel_o"] = torch.LongTensor(rel_o)
    graph.ndata["id"] = torch.from_numpy(u_id).long()
    graph.ndata["t_id"] = torch.from_numpy(t_id).long()

    # ----Calculate the position of s at each moment in the big picture at the previous moment
    s_his = defaultdict(int)  # Record the previous index
    s_his_t = defaultdict(int)  # Record the time of the previous occurrence
    s_his_f = defaultdict(int)  # Record the previous index
    s_his_l = defaultdict(int)  # Record the length of the historical sequence
    s_last_index = np.zeros(
        (nodes_num, L), dtype=int
    )  # Record the index of each moment
    s_last_t = L * np.ones(
        (nodes_num, L), dtype=int
    )  # Record the time of the last interaction at each moment. If the last interaction did not occur, it is recorded as L.
    s_last_f = graph.num_nodes() * np.ones(
        (nodes_num, L), dtype=int
    )  # Record the index of each moment. The value of the node that has not happened is nodes_num.
    s_last_l = np.zeros((nodes_num, L), dtype=int)
    node_id = graph.ndata["id"].numpy()
    time_id = graph.ndata["t_id"].numpy()
    # node_id_new = len(node_id) - 1 - np.unique(node_id[::-1], return_index=True)[1]
    # initialization
    id, node_id_f = np.unique(node_id, return_index=True)
    for n_i, ei in enumerate(id):
        s_his[ei] = node_id_f[n_i]
        s_his_t[ei] = L  # T+1 time
        s_his_f[ei] = graph.num_nodes()
        s_his_l[ei] = 0  # 0
    s_last_index[id, 0] = node_id_f
    for i, tim in enumerate(time):
        if name == "Sample":
            data_1 = get_data_with_t(data, tim)
        else:
            data_1 = data[i]
        src1, rel1, dst1 = data_1[:, [0, 1, 2]].transpose()
        # -----Find an array to record the index of each element at this moment------
        en = np.unique((src1, dst1))
        if i > 0:
            s_last_index[:, i] = s_last_index[:, i - 1]
            s_last_t[:, i] = s_last_t[:, i - 1]
            s_last_f[:, i] = s_last_f[:, i - 1]
            s_last_l[:, i] = s_last_l[:, i - 1]
        for e_i, e in enumerate(en):
            s_last_index[e, i] = s_his[e]
            s_last_t[e, i] = s_his_t[e]
            s_last_f[e, i] = s_his_f[e]
            s_last_l[e, i] = s_his_l[e]
            s_his[e] = e_i + idx[i]  # Calculate the number of the last interaction of s
            s_his_f[e] = (
                e_i + idx[i]
            )  # Calculate the number of the last interaction of s
            s_his_t[e] = i  # Calculate the time when the last interaction occurred in s
            s_his_l[e] = s_his_l[e] + 1  # Calculate the length of s historical sequence
        # -----------------------------------------
    return (
        graph,
        torch.from_numpy(s_last_index).long(),
        torch.from_numpy(s_last_t).long(),
        torch.from_numpy(s_last_f).long(),
        torch.from_numpy(s_last_l).long(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data")
    parser.add_argument("--data", default="Sample")
    args = parser.parse_args()
    print(args)

    # Load datasets
    data = load_data(args.data)

    # Number of entities, relations
    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # Train, valid, and test list
    train_list = split_by_time(data.train)
    valid_list = split_by_time(data.valid)
    test_list = split_by_time(data.test)

    # Total data = train data + valid data + test data
    total_data = train_list + valid_list + test_list

    # Number of time
    time_num = [len(np.unique(da[:, [0, 2]])) for da in total_data]
    total_times = range(len(total_data))

    # Save path
    save_path = "./data/" + os.sep + args.data + os.sep + "graph_" + args.data

    # Generat graph
    graph, s_index, s_t, s_f, s_l = generate_graph(
        total_data,
        total_times,
        time_num=time_num,
        nodes_num=num_nodes,
        rel_nums=num_rels,
        name=args.data,
    )

    # Save graph data from numpy array to file.
    save_graphs(
        save_path, graph, {"s_index": s_index, "s_t": s_t, "s_f": s_f, "s_l": s_l}
    )
