# from craft import CRAFT
import csv
import os
import subprocess

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric
from bson import json_util
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient

from data import guide_line

# from GCN.src.models.data_for_GCN import from_networkx
from GCN.src.models.final_model_test import Net
from GCN.src.pipeline.graph import Grapher

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Kết nối đến MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Med_Scan"]
collection = db["Medicines"]

# Thư mục để lưu ảnh
UPLOAD_FOLDER = "input"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


pretrained_model_path = (
    "D:/Luan Van/Project/med-scan-backend/GCN/src/models/DP/models/k3_01.pth"
)
# pretrained_model_path = "D:/Luan Van/Project/med-scan-backend/GCN/src/models/best_ChebConv_model_k_3_new_newdataset_embedFeatures_updataDataSet_updateClassWeights_noEmbedTextObject_addFeaturesDrugnameNearestnb_AddTensorFeatures_updateMedName.pth"

# Load the pre-trained GCN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
model.load_state_dict(torch.load(pretrained_model_path))
model.eval()


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


# Xử lý ảnh request và reply
@app.route("/api/extract_medicines", methods=["POST"])
def extract_medicines():
    # Đảm bảo rằng thư mục UPLOAD_FOLDER (input) tồn tại
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])

    # Xoá tất cả các tệp trong thư mục UPLOAD_FOLDER
    for file_name in os.listdir(app.config["UPLOAD_FOLDER"]):
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Không thể xoá tệp {file_path}: {e}")

    uploaded_image = request.files["image"]

    if uploaded_image:
        # Đổi tên cho hình ảnh thành "1.jpg" mặc định
        uploaded_image.filename = "1.jpg"
        # Lưu hình ảnh vào thư mục ""data""
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_image.filename)
        uploaded_image.save(image_path)

        # Gọi CRAFT và inference.py bằng subprocess
        craft_command = [
            "python",
            "CRAFT/pipeline.py",
            "--test_folder",
            app.config["UPLOAD_FOLDER"],
        ]

        inference_command = ["python", "TextDetection/TextRecognition/inference.py"]

        subprocess.run(craft_command)
        subprocess.run(inference_command)

        # Đọc đồ thị từ hóa đơn
        connect = Grapher(uploaded_image.filename.split(".")[0])
        G, _, _ = connect.graph_formation()
        df = connect.relative_distance()

        # Chuyển đồ thị thành đối tượng Data
        individual_data = from_networkx(G)

        # print("individual_data")
        # print(individual_data)

        # Xử lý đặc trưng
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
        individual_data.x = features

        # print("KIEM TRA DF")
        # print(df)

        # print(model)
        # Chạy mô hình
        model.eval()
        with torch.no_grad():
            logits = model(individual_data.x, individual_data.edge_index)
            probabilities = F.softmax(logits, dim=1)
        # print("- individual_data.x")
        # print(individual_data.x)
        # print("- individual_data.edge_index")
        # print(individual_data.edge_index)
        # print("- logits")
        # print(logits)

        text_fields = df.iloc[:, 5].tolist()

        num_medicine_name_labels = 0

        labeled_texts = []
        ##Du doan dua tren xac suat
        threshold = 0.3
        for i, text_field in enumerate(text_fields):
            probability_medicine_name = probabilities[
                i, 1
            ].item()  # Xác suất lớp "medicine_name"

            # In ra xác suất cho từng dòng
            print(f"Text: {text_field},- Probability: {probability_medicine_name}")

            # Kiểm tra xác suất và gán nhãn
            if probability_medicine_name > threshold:
                label = "undefined"
            else:
                label = "medicine_name"
                num_medicine_name_labels += 1
            labeled_texts.append({"text": text_field, "label": label})

        ## Du doan dua tren dac trung
        # for i, text_field in enumerate(text_fields):
        #     n_is_drug = df.loc[i, "n_is_drug"]
        #     similarity_scores = df.loc[i, "similarity_scores"]
        #     nearby_words = df.loc[i, "n_nearby_words"]

        #     # Kiểm tra nếu thỏa các điều kiện
        #     if n_is_drug == 1 and similarity_scores > 0.1:
        #         label = "medicine_name"
        #         num_medicine_name_labels += 1

        #     # Kiểm tra từ cột 'n_nearby_words'
        #     elif text_field in nearby_words:
        #         # Nếu text_field trùng với bất kỳ giá trị trong cột n_nearby_words
        #         label = "medicine_name"
        #         num_medicine_name_labels += 1
        #     else:
        #         label = "ud"

        #     labeled_texts.append({"text": text_field, "label": label})

        # In ra số lượng nhãn "medicine_name"
        print(f"Tổng số nhãn 'medicine_name': {num_medicine_name_labels}")

        # Trả kết quả về cho client
        response_data = {
            "labeled_texts": labeled_texts,
            "num_medicine_name_labels": num_medicine_name_labels,
        }
        return jsonify(response_data)


# Enpoint để tìm kiếm tên thuốc từ db
@app.route("/api/search_medicine", methods=["GET"])
def search_medicine():
    search_query = request.args.get("query")

    if not search_query:
        return jsonify({"message": "Vui lòng cung cấp thông tin tìm kiếm."}), 400

    # Sử dụng pymongo để truy vấn cơ sở dữ liệu và chỉ lấy trường "name"
    results = list(
        collection.find(
            {"name": {"$regex": search_query, "$options": "i"}},
            {"id": 1, "name": 1, "longDescription": 1, "_id": 0},
        )
    )
    if not results:
        return jsonify({"message": "Không tìm thấy kết quả phù hợp."}), 404
    else:
        # Chuyển đổi kết quả thành JSON sử dụng json_util
        return jsonify(json_util.dumps(results))


# Enpoint để trả về "name" và "longDescription" dựa trên "id" của loại thuốc
@app.route("/api/medicine_details/<id>", methods=["GET"])
def get_medicine_details(id):
    # Tìm kiếm loại thuốc theo "id"
    medicine = collection.find_one(
        {"id": id}, {"name": 1, "longDescription": 1, "images": 1, "_id": 0}
    )

    if medicine:
        return jsonify(medicine)
    else:
        return jsonify({"message": "Không tìm thấy loại thuốc với ID cung cấp."}), 404


# Enpoint để lấy Guided_line
@app.route("/api/guideline", methods=["GET"])
def get_guideline():
    return jsonify(guide_line)


if __name__ == "__main__":
    app.run(debug=True)
