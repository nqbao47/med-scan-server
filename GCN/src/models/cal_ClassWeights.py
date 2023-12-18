import numpy as np

# class_counts = {
#     "company": 687,
#     "address": 1677,
#     "invoice": 598,
#     "date": 624,
#     "total": 626,
#     "undefined": 29414,
# }

class_counts = {
    "medicine_name": 2301,
    "undefined": 13733,
}

total_samples = sum(class_counts.values())
num_classes = len(class_counts)

class_weights = {
    cls: total_samples / (num_classes * freq) for cls, freq in class_counts.items()
}

print("Class number distribution in the dataset:")
for cls, count in class_counts.items():
    print(f"{cls}: {count}")

print("\nWeights used to balance class distribution for NLLLoss propagation:")
print(class_weights)
