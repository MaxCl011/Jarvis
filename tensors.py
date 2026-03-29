import torch


a = torch.tensor([1.0, 2.0, 3.0])

zeros = torch.zeros(3, 4)
ones = torch.ones(3, 4)

rand = torch.rand(3, 4) 
randn = torch.randn(3, 4)

r = torch.arange(0, 10, step=2)

print(rand)
print(rand.shape)
print(rand.dtype)