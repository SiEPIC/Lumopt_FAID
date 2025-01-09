import torch

print("Is CUDA available?", torch.cuda.is_available())
print("Number of GPUs available:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("GPU device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
