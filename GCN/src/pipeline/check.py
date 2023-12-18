import numpy as np
import torch
from sklearn.utils import class_weight


def check_class_weights(class_weights):
    if len(class_weights) != 2:
        print("Error: The length of class_weights should be 2.")
    else:
        print(f"Class Weights: {class_weights}")


if __name__ == "__main__":
    # Thay thế dòng sau đây bằng class_weights thực tế của bạn
    class_weights_example = [1.0, 2.0]

    check_class_weights(class_weights_example)


## Kiểm tra các thông tin về đối  tượng có trong list để đảm bảo
# import torch
# import torch_geometric

# # Đường dẫn đến tệp data_withtexts.dataset
# save_path = "../../data/processed/"
# data = torch.load(save_path + "data_withtexts.dataset")

# # Lấy một số đối tượng từ data để kiểm tra
# # Ở đây, tôi lấy đối tượng đầu tiên từ danh sách
# example_data = data[0]

# # In thông tin về đối tượng đầu tiên
# print("Example Data Object:")
# print(f"Number of nodes: {example_data.num_nodes}")
# print(f"Number of edges: {example_data.num_edges}")
# print(f"Node features (x): {example_data.x}")
# print(f"Edge index: {example_data.edge_index}")
# print(f"Labels (y): {example_data.y}")
# print(f"Train mask: {example_data.train_mask}")
# print(f"Validation mask: {example_data.val_mask}")
# print(f"Test mask: {example_data.test_mask}")

# # Bạn có thể thêm thông tin khác nếu cần
