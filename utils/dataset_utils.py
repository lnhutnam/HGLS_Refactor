from tqdm import tqdm

import torch
import numpy as np

import utils.datasets as knwlgrh
from utils.graph_utils import add_object, add_subject


def load_all_answers(total_data: np.ndarray, num_rel: int):
    """Construct a dictionary that contains subjects for all (rel, object) queries and objects for all (subject, rel) queries

    Args:
        total_data (np.ndarray): input triples data.
        num_rel (int): number of relation in dataset.

    Returns:
        default dict: return default dict `all_objects`, and `all_subjects`
    """
    all_subjects, all_objects = {}, {}
    for line in total_data:
        s, r, o = line[:3]
        add_subject(s, o, r, all_subjects, num_rel=num_rel)
        add_object(s, o, r, all_objects, num_rel=0)
    return all_objects, all_subjects


def load_all_answers_for_filter(
    total_data: np.ndarray, num_rel: int, rel_p: bool = False
):
    """
    Construct a dictionary that contains subjects for all (rel, object) queries, and objects for all (subject, rel) queries.
    Args:
        total_data (np.ndarray): input triples data.
        num_rel (int): number of relation in dataset.
        rel_p (bool, optional): _description_. Defaults to False.

    Returns:
        default dict: all answers (rel, object) queries and objects for all (subject, rel) queries
    """

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


def load_all_answers_for_time_filter(
    total_data: np.ndarray, num_rels: int, num_nodes: int, rel_p: bool = False
):
    """_summary_

    Args:
        total_data (np.ndarray): input data
        num_rels (int): _description_
        num_nodes (int): _description_
        rel_p (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    all_ans_list = []
    all_snap = split_by_time(total_data)
    for snap in all_snap:
        all_ans_t = load_all_answers_for_filter(snap, num_rels, rel_p)
        all_ans_list.append(all_ans_t)

    return all_ans_list


def split_by_time(data: np.ndarray):
    """_summary_

    Args:
        data (np.ndarray): _description_

    Returns:
        list: list of graph snapshots
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


def load_data(dataset: str, bfs_level: int = 3, relabel: bool = False):
    """Function for loading data

    Args:
        dataset (str): input name of need loading dataset.
        bfs_level (int, optional): level for bfs graph traversal. Defaults to 3.
        relabel (bool, optional): is needed to be relabel dataset when you use general graph such as `aifb`, `mutag`, `bgs`, and `am`. Defaults to False.

    Raises:
        ValueError: unknown dataset.

    Returns:
        RGCNLinkDataset: _description_
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
