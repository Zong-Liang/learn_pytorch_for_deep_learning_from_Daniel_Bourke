# 01_fundamentals

## `torch.squeeze()`

`torch.squeeze()` 是 PyTorch 中的一个函数，用于移除张量（tensor）中维度大小为1的维度。

```python
import torch

# 移除所有大小为1的维度
# 创建一个有大小为1的维度的张量
x = torch.randn(1, 3, 1, 4, 1)
print(x.shape)  # 输出 torch.Size([1, 3, 1, 4, 1])

# 使用 squeeze 移除所有大小为1的维度
y = torch.squeeze(x)
print(y.shape)  # 输出 torch.Size([3, 4])

# 移除指定维度上的大小为1的维度
# 创建一个有大小为1的维度的张量
x = torch.randn(2, 1, 3, 1)
print(x.shape)  # 输出 torch.Size([2, 1, 3, 1])

# 使用 squeeze 移除特定维度上大小为1的维度
y = torch.squeeze(x, dim=1)  # 移除第二维上的大小为1的维度
print(y.shape)  # 输出 torch.Size([2, 3, 1])
```

## `torch.unsqueeze()`

`torch.unsqueeze()` 是 PyTorch 中的一个函数，用于在张量（tensor）中的指定位置添加一个大小为1的新维度。

```python
import torch

# 在指定位置添加一个大小为1的维度
# 创建一个张量
x = torch.randn(3, 4)
print(x.shape)  # 输出 torch.Size([3, 4])

# 使用 unsqueeze 在第二维上添加一个大小为1的维度
y = torch.unsqueeze(x, dim=1)
print(y.shape)  # 输出 torch.Size([3, 1, 4])

# 在多个位置添加大小为1的维度
# 创建一个张量
x = torch.randn(3, 4)
print(x.shape)  # 输出 torch.Size([3, 4])

# 使用 unsqueeze 在第一维和第三维上添加大小为1的维度
y = torch.unsqueeze(x, dim=(0, 2))
print(y.shape)  # 输出 torch.Size([1, 3, 4, 1])

# 在不同位置添加大小为1的维度
# 创建一个张量
x = torch.randn(3, 4)
print(x.shape)  # 输出 torch.Size([3, 4])

# 使用 unsqueeze 在第一维和第四维上分别添加大小为1的维度
y = torch.unsqueeze(x, dim=0)
y = torch.unsqueeze(y, dim=-1)
print(y.shape)  # 输出 torch.Size([1, 3, 4, 1])
```

## `torch.permute()`

`torch.permute()` 是 PyTorch 中的一个函数，用于对张量（tensor）的维度进行排列（置换），以改变张量的形状。

```python
import torch

# 创建一个示例张量
x = torch.randn(2, 3, 4)
print(x.shape)  # 输出 torch.Size([2, 3, 4])

# 使用 permute 对维度进行排列
y = x.permute(1, 2, 0)
print(y.shape)  # 输出 torch.Size([3, 4, 2])
```

> 在上面的示例中，原始张量 `x` 的维度顺序是 (2, 3, 4)，使用 `x.permute(1, 2, 0)` 对维度进行排列后，新的张量 `y` 的维度顺序变成了 (3, 4, 2)，即原来的第一个维度成为新的第二个维度，原来的第二个维度成为新的第三个维度，原来的第三个维度成为新的第一个维度。
>
> 注意，`torch.permute()` 返回一个新的张量，不会修改原始张量。

## `torch.reshape()`

`torch.reshape()` 是 PyTorch 中的一个函数，用于改变张量（tensor）的形状，即重新排列张量中的元素，但不改变元素的数量。

```python
import torch

# 创建一个示例张量
x = torch.arange(1, 13)  # 创建一个包含1到12的张量
print(x)
# 输出:
# tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

# 使用 reshape 将张量从一维变为二维
y = torch.reshape(x, (3, 4))
print(y)
# 输出:
# tensor([[ 1,  2,  3,  4],
#         [ 5,  6,  7,  8],
#         [ 9, 10, 11, 12]])

# 使用 reshape 将张量从二维变为一维
z = torch.reshape(y, (-1,))
print(z)
# 输出:
# tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
```

> 需要注意的是，新形状的元素数量必须与原始张量的元素数量一致，否则会引发错误。在使用 `torch.reshape()` 时，通常使用 `-1` 表示某个维度的大小应该根据其他维度的大小自动计算，以确保元素数量一致。
>
> 此外，`torch.reshape()` 返回一个新的张量，不会修改原始张量。