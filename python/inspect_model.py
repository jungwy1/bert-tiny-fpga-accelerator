import torch

sd = torch.load("pytorch_model.bin", map_location="cpu", weights_only=True)

# state_dict = { "텐서이름": 텐서값, ... } 형태의 딕셔너리
for name, tensor in sd.items():
    print(f"{name:60s} {tuple(tensor.shape)}")
