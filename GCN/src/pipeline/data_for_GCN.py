import os
import random

import networkx as nx
import numpy as np
import scipy.sparse
import torch
import torch_geometric.data
from graph_copy import Grapher
from transformers import DistilBertModel, DistilBertTokenizer

# Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")


def embed_text(text):
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings


def embed_objects(data):
    embeddings = []
    for text in data.text:
        text_embedding = embed_text(text)
        embeddings.append(text_embedding)

    data.embedding = torch.tensor(embeddings, dtype=torch.float32)
    data.x = torch.cat([data.x, data.embedding], dim=1)
    data.num_node_features = data.x.shape[1]

    # Thêm hàm này sau khi bạn đã tạo đối tượng `data`
    embed_objects(data)


def create_feature_vector(df):
    features = []
    for idx, row in df.iterrows():
        # Boolean features
        boolean_features = [
            row["Object"].isalpha(),
            row["Object"].islower(),
            row["Object"].isupper(),
            row["Object"].isnumeric(),
            row["Object"].isspace(),
            any(char.isdigit() or char.isalpha() for char in row["Object"]),
        ]

        # Numerical features (relative distances)
        numerical_features = [row["rd_r"], row["rd_b"], row["rd_l"], row["rd_t"]]

        # Textual features (DistilBERT embeddings)
        text_embedding = embed_text(row["Object"])

        # Concatenate all features
        node_features = np.concatenate(
            [boolean_features, numerical_features, text_embedding]
        )
        features.append(node_features)

    return np.array(features)


def from_networkx(G):
    """Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

    G = nx.convert_node_labels_to_integers(G)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[key] = [value] if i == 0 else data[key] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data["edge_index"] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()
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

    for file in all_files:
        connect = Grapher(file)
        G, _, _ = connect.graph_formation()
        df = connect.relative_distance()
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
        ]

        features = torch.tensor(df[feature_cols].values.astype(np.float32))

        for col in df.columns:
            try:
                df[col] = df[col].str.strip()
            except AttributeError:
                pass

        print(df)

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

        # Additional code for embedding features
        df["Object"] = df["Object"].astype(str)
        embeddings = create_feature_vector(df)
        individual_data.embedding = torch.tensor(embeddings, dtype=torch.float32)

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

        print(f"{file} ---> Success")
        print(df)
        list_of_graphs.append(individual_data)

    data = torch_geometric.data.Batch.from_data_list(list_of_graphs)
    data.edge_attr = None

    save_path = "../../data/test/"
    torch.save(data, save_path + "data_embedFeatures.dataset")


if __name__ == "__main__":
    get_data()
