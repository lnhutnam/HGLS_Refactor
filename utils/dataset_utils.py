from tqdm import tqdm

import torch
import numpy as np

import utils.datasets as knwlgrh
from utils.graph_utils import add_object, add_subject


def load_all_answers(total_data, num_rel):
    """store subjects for all (rel, object) queries and objects for all (subject, rel) queries

    Args:
        total_data (torch.Tensor): _description_
        num_rel (): _description_

    Returns:
        _type_: _description_
    """
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[:3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(total_data, num_rel, rel_p=False):
    """_summary_

    Args:
        total_data (_type_): _description_
        num_rel (_type_): _description_
        rel_p (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    # store subjects for all (rel, object) queries and
    # objects for all (subject, rel) queries
    def add_relation(e1, e2, r, d):
        if not e1 in d:
            d[e1] = {}
        if not e2 in d[e1]:
            d[e1][e2] = set()
        d[e1][e2].add(r)

    all_ans = {}
    for line in total_data:
        s, r, o = line[:3]
        if rel_p:
            add_relation(s, o, r, all_ans)
            add_relation(o, s, r + num_rel, all_ans)
        else:
            add_subject(s, o, r, all_ans, num_rel=num_rel)
            add_object(s, o, r, all_ans, num_rel=0)
    return all_ans


def load_all_answers_for_time_filter(total_data, num_rels, num_nodes, rel_p=False):
    """_summary_

    Args:
        total_data (_type_): _description_
        num_rels (_type_): _description_
        num_nodes (_type_): _description_
        rel_p (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    # output_label_list = []
    # for all_ans in all_ans_list:
    #     output = []
    #     ans = []
    #     for e1 in all_ans.keys():
    #         for r in all_ans[e1].keys():
    #             output.append([e1, r])
    #             ans.append(list(all_ans[e1][r]))
    #     output = torch.from_numpy(np.array(output))
    #     output_label_list.append((output, ans))
    # return output_label_list
    return all_ans_list


def split_by_time(data):
    """_summary_

    Args:
        data (_type_): _description_

    Returns:
        _type_: _description_
    """
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t represents the time when the last triple read occurred, and the triples in the data set are required to be sorted in the order of occurrence.
        if latest_t != t:  # Triplets that occur at the same time
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
        
    # Add the last snapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique(
            (snapshot[:, 0], snapshot[:, 2]), return_inverse=True
        )  # relabel
        uniq_r = np.unique(snapshot[:, 1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r) * 2)
    print(
        "# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}".format(
            np.average(np.array(nodes)),
            np.average(np.array(rels)),
            len(snapshot_list),
            max([len(_) for _ in snapshot_list]),
            min([len(_) for _ in snapshot_list]),
            max(union_num),
            min(union_num),
        )
    )
    return snapshot_list

def slide_list(snapshots, k=1):
    """_summary_

    Args:
        snapshots (torch.Tensor): all snapshot
        k (int, optional): padding K history for sequence stat. Defaults to 1.

    Yields:
        _type_: _description_
    """
    k = k  # k=1 needs to take the history of length k, and add a label of length 1
    if k > len(snapshots):
        print(
            "ERROR: history length exceed the length of snapshot: {}>{}".format(
                k, len(snapshots)
            )
        )
    for _ in tqdm(range(len(snapshots) - k + 1)):
        yield snapshots[_ : _ + k]


def load_data(dataset, bfs_level=3, relabel=False):
    """_summary_

    Args:
        dataset (_type_): _description_
        bfs_level (int, optional): _description_. Defaults to 3.
        relabel (bool, optional): _description_. Defaults to False.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if dataset in ["aifb", "mutag", "bgs", "am"]:
        return knwlgrh.load_entity(dataset, bfs_level, relabel)
    elif dataset in ["FB15k", "wn18", "FB15k-237"]:
        return knwlgrh.load_link(dataset)
    elif dataset in [
        "ICEWS18",
        "ICEWS14",
        "GDELT",
        "SMALL",
        "ICEWS14s",
        "ICEWS05-15",
        "YAGO",
        "WIKI",
    ]:
        return knwlgrh.load_from_local("./data", dataset)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def construct_snap(test_triples, num_nodes, num_rels, final_score, topK):
    """_summary_

    Args:
        test_triples (_type_): _description_
        num_nodes (_type_): _description_
        num_rels (_type_): _description_
        final_score (_type_): _description_
        topK (_type_): _description_

    Returns:
        _type_: _description_
    """
    # sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    _, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, r = test_triples[_][0], test_triples[_][1]
            if r < num_rels:
                predict_triples.append([test_triples[_][0], r, index])
            else:
                predict_triples.append([index, r - num_rels, test_triples[_][0]])

    # Convert to numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples


def construct_snap_r(test_triples, num_nodes, num_rels, final_score, topK):
    """Constructing knowledge graph snapshot 

    Args:
        test_triples (_type_): _description_
        num_nodes (_type_): _description_
        num_rels (_type_): _description_
        final_score (_type_): _description_
        topK (_type_): _description_

    Returns:
        _type_: _description_
    """
    # sorted_score, indices = torch.sort(final_score, dim=1, descending=True)
    _, indices = torch.sort(final_score, dim=1, descending=True)
    top_indices = indices[:, :topK]
    predict_triples = []
    
    for _ in range(len(test_triples)):
        for index in top_indices[_]:
            h, t = test_triples[_][0], test_triples[_][2]
            if index < num_rels:
                predict_triples.append([h, index, t])
                # predict_triples.append([t, index+num_rels, h])
            else:
                predict_triples.append([t, index - num_rels, h])
                # predict_triples.append([t, index-num_rels, h])

    # Convert to numpy array
    predict_triples = np.array(predict_triples, dtype=int)
    return predict_triples

