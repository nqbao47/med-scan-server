import argparse
import csv
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import Parameter
from torch_geometric.nn import ChebConv, GCNConv

data_path = (
    "D:/Luan Van/Project/med-scan-backend/GCN/data/test/"
    + "data_embedFeatures_updateClassWeights_addFeaturesDrugnameNearestnb_AddTensorFeatures_updateMedName.dataset"
)
data = torch.load(data_path)


# print(f"training nodes: {data.y[data.train_mask].shape}")
# print(f"validation nodes: {data.y[data.val_mask].shape}")
# print(f"testing nodes: {data.y[data.test_mask].shape}")


best_val_loss = float(
    "inf"
)  # Giả sử là vô cùng lớn để đảm bảo lần đầu chạy sẽ lưu mô hình
best_epoch = -1

model_save = (
    "D:/Luan Van/Project/med-scan-backend/GCN/src/models/DP/models/update_k3_01.pth"
)
csv_file_path = (
    "D:/Luan Van/Project/med-scan-backend/GCN/src/models/DP/logs/update_k3_01.csv"
)

with open(csv_file_path, mode="w", newline="") as csv_file:
    fieldnames = [
        "Epoch",
        "Train Loss",
        "Validation Loss",
        "Train Accuracy",
        "Validation Accuracy",
        "Test Accuracy",
    ]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    # type=str,
    # default="GCNConv",
    # help="GCN or ChebConv model" "--model",
    type=str,
    default="ChebConv",
)

parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of epochs to train."
)
parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
# parser.add_argument("--lr", type=float, default=0.2, help="Initial learning rate.")
# parser.add_argument("--lr", type=float, default=0.3, help="Initial learning rate.")
parser.add_argument("--verbose", type=int, default=1, help="print confusion matrix")
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)

parser.add_argument(
    "--early_stopping", type=int, default=50, help="Stopping criteria for validation"
)
parser.add_argument("--use_gdc", action="store_true", help="Use GDC preprocessing.")
args = parser.parse_args()


# print(f"number of nodes: {data.x.shape}")


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if args.model == "GCNConv":
            # cached = True is for transductive learning
            self.conv1 = GCNConv(data.x.shape[1], 16, cached=True)
            self.conv2 = GCNConv(16, 32, cached=True)
            self.conv3 = GCNConv(32, 64, cached=True)
            self.conv4 = GCNConv(64, 2, cached=True)

        elif args.model == "ChebConv":
            self.conv1 = ChebConv(data.x.shape[1], 16, K=3)
            self.conv2 = ChebConv(16, 32, K=3)
            self.conv3 = ChebConv(32, 64, K=3)
            self.conv4 = ChebConv(64, 2, K=3)

    # def forward(self, x, edge_index):
    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, data = Net().to(device), data.to(device)
# model.train()

# print("Number of parameters in the model:", sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
)

unique_classes_before = np.unique(data.y.numpy())
class_counts_before = {
    cls: np.sum(data.y.numpy() == cls) for cls in unique_classes_before
}
# print("Class Distribution Before SMOTE:")
# print(class_counts_before)

# Sử dụng SMOTE để oversample
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(data.x.numpy(), data.y.numpy())

# # Convert oversampled data back to PyTorch tensors
# data.x = torch.FloatTensor(X_resampled).to(device)
# data.y = torch.FloatTensor(y_resampled).to(device)

unique_classes_after = np.unique(y_resampled)
class_counts_after = {cls: np.sum(y_resampled == cls) for cls in unique_classes_after}
# print("\nClass Distribution After SMOTE:")
# print(class_counts_after)
# Tính toán trọng số lớp
class_weights = compute_class_weight(
    "balanced", classes=np.unique(y_resampled), y=y_resampled
)

class_weights = torch.FloatTensor(class_weights)


def train():
    model.train()
    optimizer.zero_grad()

    # weights = torch.FloatTensor([0.5838, 3.4841])
    # unique_labels = np.unique(data.y.numpy())
    # print("Unique labels in data.y:", unique_labels)

    # print("Sample labels in data.y:", data.y[:10].numpy())

    loss = F.nll_loss(
        model()[data.train_mask],
        data.y[data.train_mask].long() - 1,
        weight=class_weights.to(device),
    )
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test():
    model.eval()
    # F.nll_loss(model()[data.val_mask], data.y[data.val_mask])
    logits, accs = model(), []
    for mask_name, mask in data("train_mask", "val_mask", "test_mask"):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask] - 1).sum().item() / mask.sum().item()
        if args.verbose == 1:
            # confusion matrix
            if mask_name == "test_mask":
                conf_mat = confusion_matrix((data.y[mask] - 1).numpy(), pred.numpy())
                print(f"confusion_matrix: \n   {conf_mat}")
                class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
                print(class_accuracy)
        accs.append(acc)
    return accs


if __name__ == "__main__":
    # stopping criteria
    counter = 0
    for epoch in range(1, args.epochs + 1):
        loss = train()
        train_acc, val_acc, test_acc = test()
        with torch.no_grad():
            # print(model()[data.val_mask])
            loss_val = F.nll_loss(
                model()[data.val_mask], data.y[data.val_mask].long() - 1
            )
            print(model()[data.val_mask])

        # Lưu mô hình nếu validation loss giảm
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                model_save,
                # "best_gcnconv_model.pth",
            )

        log = "Epoch: {:03d}, train_loss:{:.4f}, val_loss:{:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}"
        print(log.format(epoch, loss, loss_val, train_acc, val_acc, test_acc))

        # Ghi thông số vào file CSV
        with open(csv_file_path, mode="a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(
                {
                    "Epoch": epoch,
                    "Train Loss": float(loss),
                    "Validation Loss": float(loss_val),
                    "Train Accuracy": float(train_acc),
                    "Validation Accuracy": float(val_acc),
                    "Test Accuracy": float(test_acc),
                }
            )

        # for first epoch
        if epoch == 1:
            largest_val_loss = loss_val

        # early stopping if the loss val does not improve/decrease for a number of epochs
        if loss_val >= largest_val_loss:
            counter += 1
            best_val_loss = loss_val
            if counter >= args.early_stopping:
                print(
                    f"EarlyStopping counter: validation loss did not increase for {args.early_stopping}!!"
                )
                break

# In ra thông báo và giá trị của best_epoch sau khi hoàn thành vòng lặp
print(
    f"Best ChebConv model saved at epoch {best_epoch} with validation loss: {best_val_loss}"
)
