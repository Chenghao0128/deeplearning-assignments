# PyTorch 语法与函数速查表

本文件用于记录 PyTorch 101 学习中常用的语法、函数和注意事项。

## 张量创建

| 函数 | 作用 | 示例 |
| --- | --- | --- |
| `torch.tensor(data)` | 根据 Python 数据创建张量 | `torch.tensor([[1, 2], [3, 4]])` |
| `torch.zeros(M, N)` | 创建全零张量 | `torch.zeros(2, 3)` |
| `torch.ones(M, N)` | 创建全一张量 | `torch.ones(2, 3)` |
| `torch.full(shape, value)` | 创建填充指定值的张量 | `torch.full((2, 3), 3.14)` |
| `torch.arange(start, stop, step)` | 创建等差数列张量 | `torch.arange(0, 10, 2)` |
| `torch.rand(M, N)` | 创建均匀分布随机张量 | `torch.rand(2, 3)` |

## 查看张量信息

| 语法 | 作用 |
| --- | --- |
| `x.shape` | 查看每个维度的大小 |
| `x.dim()` | 查看张量的维数 |
| `x.dtype` | 查看元素的数据类型 |
| `x.device` | 查看张量位于 CPU 还是 GPU |
| `x.item()` | 将单元素张量转换成 Python 数值 |