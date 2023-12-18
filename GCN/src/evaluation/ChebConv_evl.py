import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch_geometric.nn import ChebConv

data_path = (
    "../../data/test/"
    + "data_embedFeatures_updateClassWeights_addFeaturesDrugnameNearestnb_AddTensorFeatures_updateMedName.dataset"
)

data = torch.load(data_path)

print(f"training nodes: {data.y[data.train_mask].shape}")
print(f"validation nodes: {data.y[data.val_mask].shape}")
print(f"testing nodes: {data.y[data.test_mask].shape}")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ChebConv(data.x.shape[1], 16, K=3)
        # self.norm1 = torch.nn.LayerNorm(16)
        self.conv2 = ChebConv(16, 32, K=3)
        # self.norm2 = torch.nn.LayerNorm(32)
        self.conv3 = ChebConv(32, 64, K=3)
        # self.norm3 = torch.nn.LayerNorm(64)
        self.conv4 = ChebConv(64, 2, K=3)

    # def forward(self, x, edge_index, edge_weight):
    #     x = F.relu(self.conv1(x, edge_index, edge_weight))
    #     x = F.relu(self.conv2(x, edge_index, edge_weight))
    #     x = F.relu(self.conv3(x, edge_index, edge_weight))
    #     x = self.conv4(x, edge_index, edge_weight)
    #     return F.log_softmax(x, dim=1)

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


# Load mô hình đã lưu
model = Net()
model.load_state_dict(
    torch.load(
        "D:\Luan Van\Project\med-scan-backend\GCN\src\models\DP\models\k3_01.pth"
        # "../models/best_chebconv_model_k_3_new_newdataset_embedFeatures_updataDataSet_updateClassWeights_noEmbedTextObject_addFeaturesDrugnameNearestnb_AddTensorFeatures_updateMedName_new.pth"
    )
)
model.eval()

# Đánh giá ma trận độ chính xác trên tập kiểm thử và in ra các lớp được dự đoán
with torch.no_grad():
    logits = model()

    # Lấy xác suất của dự đoán
    probabilities = F.softmax(logits[data.val_mask], dim=1)

    # Chọn mẫu có xác suất cao nhất cho mỗi nhãn
    top_samples_medicine_name = data.val_mask[probabilities[:, 0].argmax()]
    top_samples_undefined = data.val_mask[probabilities[:, 1].argmax()]

    # Hiển thị thông tin về các mẫu được chọn
    print("Top Sample for Medicine Name:")
    print(data.x[top_samples_medicine_name])

    print("Top Sample for Undefined:")
    print(data.x[top_samples_undefined])

    # Lấy nhãn dự đoán và nhãn thực tế
    pred_labels = logits[data.val_mask].max(1)[1].cpu().numpy()
    true_labels = (data.y[data.val_mask] - 1).cpu().numpy()

    for i in range(len(true_labels)):
        print(
            f"Sample {i + 1} - True Class: {true_labels[i]}, Predicted Class: {pred_labels[i]}"
        )

    conf_mat = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(conf_mat)

    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    print("Class Accuracy:")
    print(class_accuracy)

# Đánh giá độ chính xác trên tập train và tập test
train_accs = []
test_accs = []
# # Đánh giá ma trận độ chính xác trên tập kiểm thử và in ra các lớp được dự đoán
# with torch.no_grad():
#     # logits = model(data.x, data.edge_index, data.edge_attr)
#     # pred_labels = logits.max(1)[1].cpu().numpy()
#     # true_labels = (data.y - 1).cpu().numpy()
#     logits = model()
#     pred_labels = logits[data.test_mask].max(1)[1].cpu().numpy()
#     true_labels = (data.y[data.test_mask] - 1).cpu().numpy()

#     for i in range(len(true_labels)):
#         print(
#             f"Sample {i + 1} - True Class: {true_labels[i]}, Predicted Class: {pred_labels[i]}"
#         )

#     conf_mat = confusion_matrix(true_labels, pred_labels)
#     print("Confusion Matrix:")
#     print(conf_mat)

#     class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
#     print("Class Accuracy:")
#     print(class_accuracy)

# # Đánh giá độ chính xác trên tập train và tập test
# train_accs = []
# test_accs = []

# # Vẽ biểu đồ
# plt.figure(figsize=(8, 5))
# plt.bar(
#     ["Train", "Test"], [train_acc_at_epoch, test_acc_at_epoch], color=["blue", "red"]
# )
# plt.title("Accuracy at Epoch 1640")  # Thay đổi tiêu đề nếu cần
# plt.ylabel("Accuracy")
# plt.show()
