# 03_neural_network_classification

## `torch.round()`

`torch.round()` 是 PyTorch 中的一个函数，用于将输入张量中的元素进行四舍五入取整操作。具体来说，它将每个元素的小数部分四舍五入为最接近的整数，并返回一个新的张量，其中包含四舍五入后的整数值。

```python
import torch

# 创建一个包含浮点数的张量
x = torch.tensor([1.2, 2.7, 3.5, 4.8])

# 使用 torch.round() 进行四舍五入取整
rounded_x = torch.round(x)

# 打印结果
print(rounded_x)  # tensor([1., 3., 4., 5.])
```

> 在上面的示例中，`torch.round()` 将每个元素四舍五入为最接近的整数。注意，结果张量的数据类型与输入张量的数据类型相同。如果输入张量是浮点型，那么输出也将是浮点型，但值是整数。
>
> `torch.round()` 在处理需要将浮点数转换为整数的情况时非常有用，例如在图像处理中进行像素值的四舍五入或者在回归任务中将连续的预测值转换为离散的类别。

## `nn.BCELoss()`

`nn.BCELoss()` 是 PyTorch 中的二元交叉熵损失函数（Binary Cross Entropy Loss）。这个损失函数通常用于二元分类问题，其中每个样本的标签可以是0或1。

具体来说，`nn.BCELoss()` 用于计算模型的输出与目标标签之间的损失，其计算方式如下：

假设模型的输出为 `output`，目标标签为 `target`，其中 `output` 包含了模型对每个样本属于类别 1 的预测概率，`target` 包含了实际的标签（0 或 1）。

损失的计算方式如下：

1. 对于每个样本，计算交叉熵损失，即对于类别 1 的标签（`target` 为 1）：
   - 若 `target` 为 1，则损失为 `-log(output)`。
   - 若 `target` 为 0，则损失为 `-log(1 - output)`。
2. 对所有样本的损失取平均，得到最终的损失值。

这个损失函数的目标是最小化模型的输出与实际标签之间的差距，以便训练模型使其能够更好地进行二元分类任务。

```python
import torch
import torch.nn as nn

# 创建模型的输出和目标标签
output = torch.tensor([0.8, 0.3, 0.6], requires_grad=True)
target = torch.tensor([1.0, 0.0, 1.0])

# 使用 nn.BCELoss() 计算损失
criterion = nn.BCELoss()
loss = criterion(torch.sigmoid(output), target)

# 打印损失值
print(loss.item())
```

## `nn.BCEWithLogitsLoss()`

`nn.BCEWithLogitsLoss()` 是 PyTorch 中的二元交叉熵损失函数（Binary Cross Entropy Loss with Logits）。这个损失函数通常用于二元分类问题，其中每个样本的标签可以是0或1。

与 `nn.BCELoss()` 不同，`nn.BCEWithLogitsLoss()` 不要求模型的输出经过 sigmoid 操作（概率化），而是直接接受模型的输出，通常被称为 "logits"。这意味着你可以在模型的输出中不必显式地应用 sigmoid 激活函数。

具体来说，`nn.BCEWithLogitsLoss()` 的计算方式如下：

假设模型的输出为 `output`，目标标签为 `target`，其中 `output` 包含了模型对每个样本属于类别 1 的分数（logits），`target` 包含了实际的标签（0 或 1）。

损失的计算方式如下：

1. 对于每个样本，计算交叉熵损失，即对于类别 1 的标签（`target` 为 1）：
   - 若 `target` 为 1，则损失为 `max(output, 0) - output * target + log(1 + exp(-abs(output)))`。
   - 若 `target` 为 0，则损失为 `log(1 + exp(-abs(output))) - output * target`。
2. 对所有样本的损失取平均，得到最终的损失值。

这个损失函数的目标是最小化模型的输出（logits）与实际标签之间的差距，以便训练模型使其能够更好地进行二元分类任务。

```python
import torch
import torch.nn as nn

# 创建模型的输出和目标标签
output = torch.tensor([0.8, -0.3, 0.6], requires_grad=True)
target = torch.tensor([1.0, 0.0, 1.0])

# 使用 nn.BCEWithLogitsLoss() 计算损失
criterion = nn.BCEWithLogitsLoss()
loss = criterion(output, target)

# 打印损失值
print(loss.item())
```

> 总的来说，如果你的模型的输出已经经过 sigmoid 操作，可以使用 `nn.BCELoss()`；如果模型的输出是未经过 sigmoid 操作的 logits，建议使用 `nn.BCEWithLogitsLoss()`，因为它更稳定。

## `nn.L1Loss()`

`nn.L1Loss()` 是 PyTorch 中的 L1 损失函数，也称为平均绝对误差（Mean Absolute Error，MAE）损失函数。它用于回归任务，通常用于衡量模型的预测值与实际目标值之间的绝对差距。

具体来说，`nn.L1Loss()` 计算的是模型的预测值和实际目标值之间的绝对差的平均值。这可以表示为以下数学公式：

```python
L1 Loss = (1/n) * Σ |predicted - target|
```

其中：

- `n` 是样本的数量。
- `predicted` 是模型的预测值。
- `target` 是实际的目标值。

在训练深度学习模型时，`nn.L1Loss()` 的目标是最小化预测值和目标值之间的绝对差距，以使模型的预测尽量接近实际目标。

```python
import torch
import torch.nn as nn

# 创建模型的预测值和实际目标值
predicted = torch.tensor([2.0, 3.0, 4.0, 5.0], requires_grad=True)
target = torch.tensor([1.5, 3.2, 3.8, 4.9])

# 使用 nn.L1Loss() 计算损失
criterion = nn.L1Loss()
loss = criterion(predicted, target)

# 打印损失值
print(loss.item())
```

## `nn.CrossEntropyLoss()`

`nn.CrossEntropyLoss()` 是 PyTorch 中用于多类别分类任务的损失函数。它通常用于计算模型的输出与实际类别标签之间的交叉熵损失，以训练深度学习模型。

具体来说，`nn.CrossEntropyLoss()` 用于计算模型的预测概率分布与实际类别标签之间的交叉熵损失。这个损失函数适用于多类别分类问题，其中每个样本的标签可以属于多个类别中的一个。在内部，它首先对模型的原始输出应用 softmax 函数，将输出转化为类别概率分布，然后计算交叉熵损失。

```python
import torch
import torch.nn as nn

# 创建模型的输出和目标标签
# 假设有3个类别
output = torch.tensor([[0.2, 0.7, 0.1], [0.9, 0.1, 0.0], [0.4, 0.4, 0.2]], requires_grad=True)
target = torch.tensor([1, 0, 2])  # 实际的类别标签

# 使用 nn.CrossEntropyLoss() 计算损失
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)

# 打印损失值
print(loss.item())
```

> 值得注意的是，`nn.CrossEntropyLoss()` 对于目标标签要求是类别的索引，而不是 one-hot 编码形式。模型的输出是经过 softmax 操作的概率分布，损失函数会自动将其与目标标签匹配并计算损失。此外，PyTorch 还会自动处理 softmax 操作，因此不需要手动应用 softmax 到模型的输出上。