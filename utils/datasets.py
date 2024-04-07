import os
from typing import List
import numpy as np


def _read_dictionary(filename: str) -> dict:
    """Function for reading file contains dictionary format

    Args:
        filename (str): input filename string.

    Returns:
        dict: data in file represented as in dict type.
    """
    d = {}
    with open(filename, "r+") as f:
        for line in f:
            line = line.strip().split("\t")
            d[int(line[1])] = line[0]
    return d


def _read_triplets(filename: str):
    """Function for reading triplets in a file.

    Args:
        filename (str): input filename string.

    Yields:
        str: a read line.
    """
    with open(filename, "r+") as f:
        for line in f:
            processed_line = line.strip().split("\t")
            yield processed_line


def _read_triplets_as_list(
    filename: str, entity_dict: dict, relation_dict: str, load_time: bool
) -> List:
    """Function for reading triplets in a file then procedure result as list.

    Args:
        filename (str): input filename string.
        entity_dict (dict): entities dictionary.
        relation_dict (dict): relation dictionary.
        load_time (bool): is neeced to load timestamp.

    Returns:
        list: list of triplets (or quad)
    """
    l = []
    for triplet in _read_triplets(filename):
        s = int(triplet[0])
        r = int(triplet[1])
        o = int(triplet[2])
        if load_time:
            st = int(triplet[3])
            l.append([s, r, o, st])
        else:
            l.append([s, r, o])
    return l


class RGCNLinkDataset(object):
    def __init__(self, name: str, dir: str = None):
        self.name = name
        if dir:
            self.dir = dir
            self.dir = os.path.join(self.dir, self.name)

        print(self.dir)

    def load(self, load_time=True):
        stat_path = os.path.join(self.dir, "stat.txt")
        entity_path = os.path.join(self.dir, "entity2id.txt")
        relation_path = os.path.join(self.dir, "relation2id.txt")

        train_path = os.path.join(self.dir, "train.txt")
        valid_path = os.path.join(self.dir, "valid.txt")
        test_path = os.path.join(self.dir, "test.txt")

        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)

        self.train = np.array(
            _read_triplets_as_list(train_path, entity_dict, relation_dict, load_time)
        )
        self.valid = np.array(
            _read_triplets_as_list(valid_path, entity_dict, relation_dict, load_time)
        )
        self.test = np.array(
            _read_triplets_as_list(test_path, entity_dict, relation_dict, load_time)
        )

        with open(stat_path, "r") as f:
            line = f.readline()
            num_nodes, num_rels, _ = line.strip().split("\t")
            num_nodes = int(num_nodes)
            num_rels = int(num_rels)

        self.num_nodes = num_nodes
        if num_nodes == len(entity_dict):
            self.num_nodes = num_nodes
            print("# Sanity Check:  entities: {}".format(self.num_nodes))
        else:
            raise ValueError("Number of entities do not match.")

        if num_rels == len(relation_dict):
            self.num_rels = num_rels
            print("# Sanity Check:  relations: {}".format(self.num_rels))
        else:
            raise ValueError("Number of relations do not match.")

        self.relation_dict = relation_dict
        self.entity_dict = entity_dict
        print("# Sanity Check:  edges: {}".format(len(self.train)))


def load_data(dataset: str) -> RGCNLinkDataset:
    """

    Args:
        dataset (str): input dataset name.

    Raises:
        ValueError: raise error if dataset not supported
        ValueError: raise error if dataset not supported

    Returns:
        RGCNLinkDataset: class object for loading dataset.
    """
    if dataset in ["FB15k", "wn18", "FB15k-237"]:
        raise ValueError("This project does not support dataset: {}".format(dataset))
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
        return load_from_local("data", dataset)
    else:
        raise ValueError("Unknown dataset: {}".format(dataset))


def load_from_local(dir: str, dataset: str) -> RGCNLinkDataset:
    """Function for loading dataset from local directory.

    Args:
        dir (str): stored path of dataset
        dataset (str): name of dataset.

    Returns:
        RGCNLinkDataset: class object for loading dataset.
    """
    data = RGCNLinkDataset(dataset, dir)
    data.load()
    return data
