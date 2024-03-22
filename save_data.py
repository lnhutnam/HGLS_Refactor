import os
import datetime
import warnings
import argparse

import dgl
from dgl.data.utils import save_graphs
from torch.utils.data import DataLoader

import numpy as np

from utils.dataset_utils import (
    load_data,
    load_all_answers_for_time_filter,
    split_by_time,
)
from utils.graph_utils import GraphDatasetOnline, CollateOnline


warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save_data")
    parser.add_argument("--data", default="Sample", help="data name: sample")
    parser.add_argument("--max_length", type=int, default=10, help="max_length")
    parser.add_argument("--max_batch", type=int, default=100, help="max_length")
    parser.add_argument("--no_batch", action="store_true", help="no_batch")
    parser.add_argument("--k_hop", type=int, default=2, help="max hop")
    parser.add_argument("--encoder", default="regcn")
    parser.add_argument("--decoder", default="rgat_r1")
    parser.add_argument("--gpu", default="0", help="gpu")
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    print("loading graph data")

    data = load_data(args["dataset"])

    # Number of entities, relations
    num_nodes = data.num_nodes
    num_rels = data.num_rels

    # Train, valid, and test list
    train_list = split_by_time(data.train)
    valid_list = split_by_time(data.valid)
    test_list = split_by_time(data.test)

    total_data = train_list + valid_list + test_list
    time_num = [
        len(np.unique(da[:, [0, 2]])) for da in total_data
    ]  # The number of entities appearing at each moment
    total_times = range(len(total_data))
    time_idx = np.zeros(len(time_num) + 1, dtype=int)
    time_idx[1:] = np.cumsum(time_num)  # Numbers at different times

    # Calculate starting IDs for different datasets
    train_sid = 0
    valid_sid = len(train_list)
    test_sid = len(valid_list) + valid_sid
    all_ans_list_test = load_all_answers_for_time_filter(
        data.test, num_rels, num_nodes, False
    )
    all_ans_list_r_test = load_all_answers_for_time_filter(
        data.test, num_rels, num_nodes, True
    )
    all_ans_list_valid = load_all_answers_for_time_filter(
        data.valid, num_rels, num_nodes, False
    )
    all_ans_list_r_valid = load_all_answers_for_time_filter(
        data.valid, num_rels, num_nodes, True
    )
    graph, data_dic = dgl.load_graphs(
        "./data" + os.sep + args["dataset"] + os.sep + "graph_" + args["dataset"]
    )
    graph = graph[0]
    node_id_new = data_dic["s_index"]
    s_t = data_dic["s_t"]
    s_f = data_dic["s_f"]
    s_l = data_dic["s_l"]

    train_set = GraphDatasetOnline(
        train_list,
        max_batch=args.max_batch,
        start_id=train_sid,
        no_batch=True,
        mode="train",
    )
    val_set = GraphDatasetOnline(
        valid_list, max_batch=100, start_id=valid_sid, no_batch=True, mode="test"
    )
    test_set = GraphDatasetOnline(
        test_list,
        max_batch=args.max_batch,
        start_id=test_sid,
        no_batch=True,
        mode="test",
    )
    co = CollateOnline(
        num_nodes,
        num_rels,
        s_f,
        s_t,
        len(total_data),
        args.data,
        args.encoder,
        args.decoder,
        max_length=args.max_length,
        all=False,
        graph=graph,
        k=args.k_hop,
    )
    train_dataset = DataLoader(
        dataset=train_set,
        batch_size=1,
        collate_fn=co.collate_rel,
        shuffle=True,
        pin_memory=True,
        num_workers=6,
    )
    val_dataset = DataLoader(
        dataset=val_set,
        batch_size=1,
        collate_fn=co.collate_rel,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )
    test_dataset = DataLoader(
        dataset=test_set,
        batch_size=1,
        collate_fn=co.collate_rel,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    train_path = "data/" + "_" + args["dataset"] + "/train/"
    valid_path = "data/" + "_" + args["dataset"] + "/val/"
    test_path = "data/" + "_" + args["dataset"] + "/test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    print(
        "Start loading train set: ",
        datetime.datetime.now(),
        "=============================================",
    )
    for i, train_data_list in enumerate(train_dataset):
        save_graphs(
            train_path + str(i) + "_" + "bin",
            [train_data_list.pop("sub_e_graph"), train_data_list.pop("sub_d_graph")],
            train_data_list,
        )
    print(
        "Start loading validation set: ",
        datetime.datetime.now(),
        "=============================================",
    )
    for i, val_data_list in enumerate(val_dataset):
        save_graphs(
            valid_path + str(i) + "_" + "bin",
            [val_data_list.pop("sub_e_graph"), val_data_list.pop("sub_d_graph")],
            val_data_list,
        )
    print(
        "Start loading test set: ",
        datetime.datetime.now(),
        "=============================================",
    )
    for i, test_data_list in enumerate(test_dataset):
        save_graphs(
            test_path + str(i) + "_" + "bin",
            [test_data_list.pop("sub_e_graph"), test_data_list.pop("sub_d_graph")],
            test_data_list,
        )
    print(
        "end", datetime.datetime.now(), "============================================="
    )
