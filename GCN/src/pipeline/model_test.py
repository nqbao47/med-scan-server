import torch

# Load the model
model_path = "../models/best_chebconv_model_k_3.pth"
model = torch.load(model_path)

# Print the model architecture
print(model)

# If the model is a neural network with layers, you can print details of each layer
for name, param in model.named_parameters():
    print(f"Layer: {name}, Size: {param.size()}, Parameters: {param.requires_grad}")

# You can also print the state_dict of the model
print("Model State Dictionary:")
print(model.state_dict())
