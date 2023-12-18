import os
import random

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from graph_copy import Grapher

# from ..pipeline.graph import Grapher


def from_networkx(G):
    """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    # Chuyển đổi đặc trưng của đỉnh thành tensor
    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    # Chuyển đổi đặc trưng của cạnh thành tensor
    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    # Chuyển đổi nhãn của đỉnh thành tensor
    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data["edge_index"] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
    # print("DATA IN G")
    # print(data)
    return data


def get_data():
    """
    returns one big graph with unconnected graphs with the following:
    - x (Tensor, optional) – Node feature matrix with shape [num_nodes, num_node_features]. (default: None)
    - edge_index (LongTensor, optional) – Graph connectivity in COO format with shape [2, num_edges]. (default: None)
    - edge_attr (Tensor, optional) – Edge feature matrix with shape [num_edges, num_edge_features]. (default: None)
    - y (Tensor, optional) – Graph or node targets with arbitrary shape. (default: None)
    - validation mask, training mask and testing mask
    """
    path = "../../Training/data/main_data/raw/box/"
    l = os.listdir(path)
    files = [x.split(".")[0] for x in l]
    files.sort()
    all_files = files

    list_of_graphs = []

    r"""to create train,test,val data"""
    files = all_files.copy()
    random.shuffle(files)

    r"""Resulting in 300 receipts for training, 60 receipts for validation, and 60 for testing."""
    training, testing, validating = files[:300], files[300:360], files[360:]

    class_counts = {}
    total_undefined = 0
    total_medicine_name = 0

    for file in all_files:
        connect = Grapher(file)
        G, _, _ = connect.graph_formation()
        df = connect.relative_distance()
        # print("dataframe in GRAPH")
        # print(df)

        individual_data = from_networkx(G)

        # print(df.columns)
        # print(df)

        feature_cols = [
            "rd_b",
            "rd_r",
            "rd_t",
            "rd_l",
            "line_number",
            "n_upper",
            "n_alpha",
            "n_spaces",
            "n_numeric",
            "n_special",
            "n_width",
            "n_height",
            "n_aspect_ratio",
            "n_quantity_related",
            "n_is_drug",
            "similarity_scores",
        ]

        features = torch.tensor(df[feature_cols].values.astype(np.float32))

        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass

        # print("DataFrame before final")
        # print(df)

        df["labels"] = df["labels"].fillna("undefined")
        df.loc[df["labels"] == "medicine_name", "num_labels"] = 1
        df.loc[df["labels"] == "undefined", "num_labels"] = 2
        # print(df[df["num_labels"].isnull()])

        assert (
            df["num_labels"].isnull().values.any() == False
        ), f"labeling error! Invalid label(s) present in {file}.csv"
        labels = torch.tensor(df["num_labels"].values.astype(int))
        text = df["Object"].values

        individual_data.x = features
        individual_data.y = labels
        individual_data.text = text

        r"""Create masks"""
        if file in training:
            individual_data.train_mask = torch.tensor([True] * df.shape[0])
            individual_data.val_mask = torch.tensor([False] * df.shape[0])
            individual_data.test_mask = torch.tensor([False] * df.shape[0])

        elif file in validating:
            individual_data.train_mask = torch.tensor([False] * df.shape[0])
            individual_data.val_mask = torch.tensor([True] * df.shape[0])
            individual_data.test_mask = torch.tensor([False] * df.shape[0])
        else:
            individual_data.train_mask = torch.tensor([False] * df.shape[0])
            individual_data.val_mask = torch.tensor([False] * df.shape[0])
            individual_data.test_mask = torch.tensor([True] * df.shape[0])

        # Print class distribution for each file
        class_counts[file] = df["labels"].value_counts().to_dict()
        print(f"Class distribution for {file}: {class_counts[file]}")

        # Accumulate totals
        total_undefined += class_counts[file].get("undefined", 0)
        total_medicine_name += class_counts[file].get("medicine_name", 0)

        # print("DataFrame final")
        # print(df)

        # print("tensor features")
        # print(features)

        print(f"{file} ---> Success")
        list_of_graphs.append(individual_data)

    # Print overall class distribution
    print("Overall class distribution:")
    print(f"Total undefined: {total_undefined}")
    print(f"Total medicine_name: {total_medicine_name}")

    data = torch_geometric.data.Batch.from_data_list(list_of_graphs)
    data.edge_attr = None

    save_path = "../../data/test/"
    torch.save(
        data,
        save_path
        + "data_embedFeatures_updateClassWeights_addFeaturesDrugnameNearestnb_AddTensorFeatures_updateMedName.dataset",
    )


if __name__ == "__main__":
    get_data()
