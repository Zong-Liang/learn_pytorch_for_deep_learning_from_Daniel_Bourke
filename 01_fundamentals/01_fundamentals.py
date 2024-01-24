import time

import torch
import numpy as np
from pprint import pprint, pformat

# Format a Python object into a pretty-printed representation.
print(pformat(torch.__version__))
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print("-" * 80)

# for reproducible, you can set torch.manual_seed(42)
print("# for reproducible, you can set torch.manual_seed(42)")
torch.manual_seed(42)
print("-" * 80)

# Represent an image as a tensor with shape [3, 224, 224] - [colour_channels, height, width]
# as in the image has 3 colour channels (red, green, blue), a height of 224 pixels and a width of 224 pixels).
# https://pytorch.org/docs/stable/tensors.html

# scalar
print("# scalar: ")
sc = torch.tensor(7)
print(sc)
print(sc.dim())
print(sc.ndim)
print(sc.size())
print(sc.shape)
print(sc.item())
print("-" * 80)

# vector
print("# vector: ")
ve = torch.tensor([1, 2])
print(ve.ndim)
print(ve.shape)
print(ve[0].item())
print(ve[1].item())
print("-" * 80)

# matrix
print("# matrix: ")
ma = torch.tensor([[1, 2]])
print(ma.ndim)
print(ma.shape)
ma = torch.tensor([[1, 2], [3, 4]])
print(ma.ndim)
print(ma.shape)
print("-" * 80)

# tensor
print("# tensor: ")
te = torch.tensor([[[1, 2]]])
print(te.ndim)
print(te.shape)
te = torch.tensor([[[1, 2], [3, 4]]])
print(te.ndim)
print(te.shape)
te = torch.tensor([[[1, 2]], [[3, 4]]])
print(te.ndim)
print(te.shape)
print("-" * 80)

# random tensors
print("# random tensors: ")
rd_te = torch.rand(size=(2, 1, 2))
print(rd_te)
print(rd_te.ndim)
print(rd_te.shape)
print(rd_te.dtype)
print("-" * 80)

# zeros
print("# zeros: ")
ze = torch.zeros(size=(3, 4))
print(ze)
print(ze.ndim)
print(ze.shape)
print(ze.dtype)
print("-" * 80)

# ones
print("# ones: ")
on = torch.ones(size=(3, 4))
print(on)
print(on.ndim)
print(on.shape)
print(on.dtype)
print("-" * 80)

# a range of tensors
print("# a range of tensors: ")
rg_te = torch.arange(0, 10, 1)
print(rg_te)
print(rg_te.ndim)
print(rg_te.shape)
print(rg_te.dtype)
print("-" * 80)

# tensor_like tensors
print("# tensor_like tensors: ")
ze_like = torch.zeros_like(rg_te)
print(ze_like)
print(ze_like.ndim)
print(ze_like.shape)
print(ze_like.dtype)
print("-" * 80)

# tensor datatype
# https://pytorch.org/docs/stable/tensors.html#data-types
print("# tensor datatype: ")
fl_32 = torch.tensor([1, 2], dtype=torch.float32, device=None, requires_grad=False)
print(fl_32)
print(fl_32.ndim)
print(fl_32.shape)
print(fl_32.dtype)
print(fl_32.device)
print(fl_32.requires_grad)
fl_16 = torch.tensor([3, 4], dtype=torch.half)
print(fl_16.dtype)
fl_64 = torch.tensor([3, 4], dtype=torch.double)
print(fl_64.dtype)
print("-" * 80)

# manipulate tensors
print("# manipulate tensors: ")
mp_te = torch.tensor([1, 2])
print(f"original tensor: {mp_te}")
print(f"+: {mp_te + 1}")
print(f"-: {mp_te - 1}")
print(f"*: {mp_te * 2}")
print(f"/: {mp_te / 2}")
print(f"element wise: {mp_te * mp_te}")
# print(f'{mp_te @ mp_te}')
print(f"matrix multiplication: {torch.matmul(mp_te, mp_te)}")
print("-" * 80)

# three most common errors in deep learning
print("# three most common errors in deep learning: ")
# tensor type issue
print("# 1. tensor type issue: ")
ts_A = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.double)
print(ts_A.dtype)
ts_B = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
print(ts_B.dtype)
# torch.matmul(ts_A, ts_B)  # expected scalar type Double but found Float
# tensor shape issue
print("# 2. tensor shape issue: ")
ts_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(ts_A.shape)
ts_B = torch.tensor([[7, 8], [9, 10], [11, 12]])
print(ts_B.shape)
# torch.matmul(ts_A, ts_B)  # mat1 and mat2 shapes cannot be multiplied (3x2 and 3x2)
# tensor device issue
print("# 3. tensor device issue: ")
ts_A = torch.tensor([[1, 2], [3, 4], [5, 6]]).cuda()
print(ts_A.device)
ts_B = torch.tensor([[7, 8, 9], [10, 11, 12]])
print(ts_B.device)
# torch.matmul(ts_A, ts_B)  # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
print("-" * 80)

# aggregation
print("# aggregation: ")
X = torch.arange(0, 100, 10).type(torch.float32)
print(X)
print(f"max: {X.max()}")
print(f"min: {X.min()}")
print(f"sum: {X.sum()}")
print(f"mean: {X.mean()}")
print(f"positional max: {X.argmax()}")
print(f"positional min: {X.argmin()}")
print("-" * 80)

# change tensors
print("# change tensors: ")
print("1. reshape tensors: ")
ts_A = torch.tensor([[1, 2], [3, 4], [5, 6]])
reshaped_ts_A = torch.reshape(ts_A, (2, 3))
print(f"before reshape: \n{ts_A}, \n{ts_A.shape}")
print(f"after reshape: \n{reshaped_ts_A}, \n{reshaped_ts_A.shape}")
print("2. view of tensors: ")
view_of_ts_A = ts_A.view(2, 3)
print(f"view_of_ts_A: \n{view_of_ts_A}")
print(f"before change view: \n{ts_A}")
view_of_ts_A[:, 0] = 666
print(f"after change view: \n{ts_A}")
print("stack tensors: ")
X = torch.tensor([1, 2, 3])
stacked_ts_A = torch.stack([X, X, X, X], dim=0)
print(f"stack along dim0: \n{stacked_ts_A}")
stacked_ts_A = torch.stack([X, X, X, X], dim=1)
print(f"stack along dim1: \n{stacked_ts_A}")
print("squeeze tensors: ")
X = torch.tensor([[1, 2, 3]])
squeezed_X = torch.squeeze(X)
print(f"before squeeze: {X}, \n{X.shape}")
print(f"after squeeze: {squeezed_X}, \n{squeezed_X.shape}")
print("unsqueeze tensors: ")
X = torch.tensor([4, 5, 6])
unsqueezed_X = torch.unsqueeze(X, dim=0)
print(f"before unsqueeze: {X}, \n{X.shape}")
print(f"after unsqueeze along dim0: \n{unsqueezed_X}, \n{unsqueezed_X.shape}")
unsqueezed_X = torch.unsqueeze(X, dim=1)
print(f"after unsqueeze along dim1: \n{unsqueezed_X}, \n{unsqueezed_X.shape}")
unsqueezed_X = torch.unsqueeze(X, dim=-2)
print(f"after unsqueeze along dim-2: \n{unsqueezed_X}, \n{unsqueezed_X.shape}")
unsqueezed_X = torch.unsqueeze(X, dim=-1)
print(f"after unsqueeze along dim-1: \n{unsqueezed_X}, \n{unsqueezed_X.shape}")
print("permute tensors: ")
X = torch.rand([3, 224, 224])
print(f"before permute: \n{X}, \n{X.shape}")
# permuted_X = torch.permute(X, (1, 2, 0))
permuted_X = X.permute(1, 2, 0)
print(f"after permute: \n{permuted_X}, \n{permuted_X.shape}")
print("-" * 80)

# indexing (selecting data from tensors)
X = torch.arange(1, 10).reshape(1, 3, 3)
print(X)
print(X[0])
print(X[0, 0])
print(X[0, 0, 0])
print(X[:, :, 0])
print(X[:, 0, :])
print(X[0, :, :])
print("-" * 80)

# pytorch tensor and numpy array
print("# pytorch tensor and numpy array:")
print("1.array to tensor: ")
arr = np.arange(1.0, 10.0, 2)
print(arr, arr.dtype)
ts = torch.from_numpy(arr)
# ts = torch.from_numpy(arr).type(torch.float32)
print(ts)
print("2. tensor to array: ")
ts = torch.arange(0.0, 10.0, 2)
print(ts, ts.dtype)
arr = ts.numpy()
print(arr, arr.dtype)
print("-" * 80)

# test reproducibility
print("# test reproducibility: ")
torch.manual_seed(42)
rd_te_A = torch.rand(3, 3)
print(rd_te_A)
torch.manual_seed(42)
rd_te_B = torch.rand(3, 3)
print(rd_te_B)
print(rd_te_A == rd_te_B)
print("-" * 80)

# move tensors on device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("# move tensors on device: ")
X = torch.rand([3, 224, 224])
print(X.device)
print("1. to gpu: ")
X = X.to(device)
print(X.device)
print("2. back to cpu: ")
X = X.cpu()
print(X.device)
print("-" * 80)
