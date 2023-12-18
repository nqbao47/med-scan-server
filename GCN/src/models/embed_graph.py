# data_for_GCN.py
import os
import random

import networkx as nx
import numpy as np
import torch
import torch_geometric.data
from transformers import DistilBertModel, DistilBertTokenizer

from ..pipeline.graph import Grapher

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

    # Thêm đặc trưng nhúng vào `data.x`
    data.x = torch.tensor(embeddings, dtype=torch.float32)


def get_data_for_inference(file_path):
    """
    Trả về dữ liệu chuẩn bị cho quá trình triển khai.

    Args:
        file_path (str): Đường dẫn đến tệp dữ liệu đồ thị cần xử lý.
    Returns:
        data (torch_geometric.data.Data): Đối tượng dữ liệu PyG.
    """
    connect = Grapher(file_path)
    G, _, _ = connect.graph_formation()
    df = connect.relative_distance()

    individual_data = from_networkx(G)

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

    df["labels"] = df["labels"].fillna("undefined")
    df.loc[df["labels"] == "medicine_name", "num_labels"] = 1
    df.loc[df["labels"] == "undefined", "num_labels"] = 2

    assert (
        df["num_labels"].isnull().values.any() == False
    ), f"labeling error! Invalid label(s) present in {file_path}.csv"
    labels = torch.tensor(df["num_labels"].values.astype(int))
    text = df["Object"].values

    individual_data.x = features
    individual_data.y = labels
    individual_data.text = text

    df["Object"] = df["Object"].astype(str)
    embeddings = create_feature_vector(df)
    individual_data.embedding = torch.tensor(embeddings, dtype=torch.float32)

    return individual_data


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


if __name__ == "__main__":
    # Thay thế 'your_file_path_here' bằng đường dẫn thực tế đến tệp dữ liệu đồ thị cần xử lý
    file_path = "your_file_path_here"
    data = get_data_for_inference(file_path)

    # Kiểm tra dữ liệu
    print(data)
