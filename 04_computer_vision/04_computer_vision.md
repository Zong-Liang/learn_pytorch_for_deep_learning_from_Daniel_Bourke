# 04_computer_vision

## `datasets.FashionMNIST()`

FashionMNIST 是一个包含时尚服饰类别的图像数据集，通常用于图像分类任务。

```python
from torchvision import datasets

# 指定数据集的存储位置和下载选项
# root 是数据集存储的文件夹路径，train 表示是否下载训练集，download 表示是否自动下载数据集
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True)

# 访问数据集的样本和标签
# 通过索引来访问特定样本，例如访问第一个训练样本
sample_image, label = train_dataset[0]
print(f"Label: {label}")

# FashionMNIST 数据集包含的类别标签是以数字表示的，你可以创建一个标签到类别名称的映射
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(f"Class Name: {class_names[label]}")
```

## `torch.eq(y_true, y_pred).sum().item()`

`torch.eq(y_true, y_pred)` 是 PyTorch 中用于比较两个张量 `y_true` 和 `y_pred` 对应元素是否相等的操作。这个操作返回一个布尔值张量，其中相等的元素为 `True`，不相等的元素为 `False`。通常，这个操作用于计算模型的预测值是否与实际目标值匹配。

`sum()` 函数用于计算布尔值张量中 `True` 的数量，即相等的元素的数量。最后，`item()` 方法将结果张量的值提取为 Python 中的标量值。

所以，`torch.eq(y_true, y_pred).sum().item()` 的目的是计算两个张量 `y_true` 和 `y_pred` 中相等元素的数量。

这个操作通常在评估分类模型的性能时用于计算正确分类的样本数量，例如计算准确率（Accuracy）。