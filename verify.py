import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())        # True on Windows with GPU
print("MPS available:", torch.backends.mps.is_available()) # True on M2 Mac

# Create a simple tensor (the core data type in PyTorch)
x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Shape:", x.shape)
print("Dtype:", x.dtype)