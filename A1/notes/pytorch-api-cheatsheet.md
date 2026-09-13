# PyTorch 101 语法与函数速查表

本表整理 Assignment 1 的 PyTorch 101 主线内容，用来快速回忆“怎么写”。概念原理、做题过程和错误分析请参阅 [`pytorch101-notes.md`](./pytorch101-notes.md)。

## 1. 导入与基本概念

```python
import torch
from torch import Tensor
```

| 名称 | 含义 | 示例 |
| --- | --- | --- |
| Tensor（张量） | PyTorch 存储和计算数据的基本对象 | 标量、向量、矩阵和更高维数组 |
| rank / dimension | 张量有多少个轴 | `(3, 2)` 是二维张量 |
| shape | 每个轴分别有多大 | `(3, 2)` 表示 3 行 2 列 |
| axis / dim | 形状中某一维的位置，从 0 开始编号 | `(B, N, M)` 中 `dim=0` 是批次轴 |

## 2. 创建张量

| 函数 | 作用 | 示例 | 注意事项 |
| --- | --- | --- | --- |
| `torch.tensor(data)` | 根据 Python 数据创建张量 | `torch.tensor([[1, 2], [3, 4]])` | 嵌套列表整体是一个参数 |
| `torch.zeros(M, N)` | 创建全零张量 | `torch.zeros(2, 3)` | 默认 `float32` |
| `torch.ones(M, N)` | 创建全一张量 | `torch.ones(2, 3)` | 形状为 `(2, 3)` |
| `torch.full(shape, value)` | 用指定值填满张量 | `torch.full((2, 3), 3.14)` | `shape` 通常写成元组 |
| `torch.eye(N)` | 创建单位矩阵 | `torch.eye(3)` | 形状为 `(3, 3)` |
| `torch.rand(M, N)` | 创建均匀随机张量 | `torch.rand(2, 3)` | 元素来自 `[0, 1)` |
| `torch.randn(M, N)` | 创建标准正态随机张量 | `torch.randn(2, 3)` | 均值约为 0，标准差约为 1 |
| `torch.arange(start, stop, step)` | 创建等差数列 | `torch.arange(0, 10, 2)` | 不包含 `stop` |
| `torch.empty((0,))` | 创建空的一维张量 | `torch.empty((0,))` | 形状为 `(0,)` |
| `torch.zeros_like(x)` | 创建与 `x` 同形状、同类型的全零张量 | `torch.zeros_like(x)` | 通常也在同一设备上 |
| `x.new_zeros(M, N)` | 以 `x` 的类型和设备创建全零张量 | `x.new_zeros(4, 5)` | 形状可以与 `x` 不同 |

指定数据类型：

```python
x = torch.zeros(2, 3, dtype=torch.float64)
```

`torch.arange` 的终点默认不包含在结果中：

```python
torch.arange(10, 30, 10)  # tensor([10, 20])
torch.arange(10, 31, 10)  # tensor([10, 20, 30])
```

## 3. 查看张量信息

假设 `x = torch.tensor([[1, 2, 3], [4, 5, 6]])`：

| 语法 | 作用 | 结果 |
| --- | --- | --- |
| `x.shape` | 查看形状 | `torch.Size([2, 3])` |
| `x.shape[0]` | 第 0 维大小 | `2` |
| `x.shape[1]` | 第 1 维大小 | `3` |
| `x.dim()` | 查看维数 | `2` |
| `x.dtype` | 查看元素类型 | `torch.int64` |
| `x.device` | 查看所在设备 | 通常为 `cpu` 或 `cuda:0` |
| `x.tolist()` | 转换成 Python 嵌套列表 | `[[1,2,3],[4,5,6]]` |
| `x[0, 1].item()` | 单元素张量转成 Python 数值 | `2` |

`.item()` 只适用于恰好包含一个元素的张量。

## 4. 数据类型与转换

| dtype | 常见用途 |
| --- | --- |
| `torch.float32` | 深度学习参数、输入和激活值的常用类型 |
| `torch.float64` | 需要更高精度的浮点计算 |
| `torch.int64` | 类别编号和张量索引 |
| `torch.bool` | 布尔掩码 |
| `torch.float16` | GPU 混合精度计算，本次作业不要求掌握 |

| 写法 | 作用 |
| --- | --- |
| `x.float()` | 转成 `torch.float32` |
| `x.double()` | 转成 `torch.float64` |
| `x.long()` | 转成 `torch.int64` |
| `x.to(torch.float32)` | 转成指定 dtype |
| `x.to(other)` | 转成与 `other` 相同的 dtype 和 device |

转换通常返回新张量，所以要接住返回值：`x = x.float()`。

## 5. 普通索引与切片

二维张量使用 `x[行索引, 列索引]`。切片格式为 `start:stop:step`：包含 `start`，不包含 `stop`；省略开头或结尾表示从头开始或取到末尾。

假设 `x.shape == (M, N)`：

| 写法 | 作用 | 输出形状 |
| --- | --- | --- |
| `x[0, 1]` | 第 1 行、第 2 列的一个元素 | `()` |
| `x[0]` 或 `x[0, :]` | 第一行 | `(N,)` |
| `x[-1, :]` | 最后一行 | `(N,)` |
| `x[:, 2]` | 第三列，整数索引消除列轴 | `(M,)` |
| `x[:, 2:3]` | 第三列，切片保留列轴 | `(M, 1)` |
| `x[:2, :3]` | 前两行、前三列 | `(2, 3)` |
| `x[::2, 1::2]` | 偶数编号行、奇数编号列 | 二维 |
| `x[:, -3:]` | 最后三列 | `(M, 3)` |

这里的偶数行是索引 `0, 2, 4, ...`，奇数列是索引 `1, 3, 5, ...`。需要满足题目指定形状时，要先判断该用整数索引还是切片。

## 6. 原地修改、视图与复制

```python
x[0, 1] = 10       # 修改一个元素
x[:2, 2:6] = 2     # 修改一个切片
x *= 2              # 原地乘法
```

这些写法都会直接改变原张量 `x`。若题目要求输入不能被修改：

```python
y = x.clone()
y[0, 0] = 0
```

切片通常是共享原数据的视图，修改 `y = x[:2, :2]` 可能同时修改 `x`。需要独立数据时使用 `y = x[:2, :2].clone()`。

## 7. Python 循环与 `zip`

遍历形状并计算元素总数：

```python
num_elements = 1
for dimension in x.shape:
    num_elements *= dimension
```

将坐标和值一一配对：

```python
for index, value in zip(indices, values):
    x[index] = value
```

若 `indices = [(0,0), (1,0)]`，`values = [4,5]`，`zip` 依次产生 `((0,0),4)` 和 `((1,0),5)`。虽然 `values` 是一维列表，每个数仍可与一个二维坐标配对。

## 8. 整数张量高级索引

```python
rows = torch.tensor([1, 0, 3])
cols = torch.tensor([0, 1, 2])
y = x[rows, cols]
```

它取得 `x[1,0]`、`x[0,1]`、`x[3,2]`。两组索引是对应配对，不是所有行列组合。

### One-hot 编码

```python
labels = [1, 4, 3, 2]
N = len(labels)
C = max(labels) + 1
y = torch.zeros((N, C), dtype=torch.float32)
rows = torch.arange(N)
y[rows, labels] = 1
```

`N` 是样本数；索引从 0 开始，因此类别列数必须是 `max(labels) + 1`。

## 9. 布尔索引

```python
mask = x > 0
positive_values = x[mask]
positive_sum = x[x > 0].sum().item()
```

比较会逐元素产生与 `x` 同形状的布尔张量；`True` 的位置被选出。最后的写法对 `x` 只有一次索引操作。

## 10. 改变形状与交换轴

改变形状不能改变元素总数。

| 写法 | 作用 | 注意事项 |
| --- | --- | --- |
| `x.view(new_shape)` | 以新形状查看相同数据 | 通常要求内存连续 |
| `x.reshape(new_shape)` | 改变形状 | 必要时会复制，更灵活 |
| `x.view(-1)` | 展平成一维 | `-1` 由 PyTorch 自动推算 |
| `x.view(1, -1)` | 变成一行 | 形状为 `(1, 元素总数)` |
| `x.view(-1, 1)` | 变成一列 | 形状为 `(元素总数, 1)` |
| `x.t()` | 转置二维矩阵 | 只适合二维张量 |
| `x.transpose(a, b)` | 交换两个轴 | 其他轴顺序不变 |
| `x.permute(...)` | 重新排列所有轴 | 必须给出每个轴的新顺序 |
| `x.contiguous()` | 得到内存连续布局 | 常在换轴后、`view` 前使用 |

如果 `x` 有 24 个元素：`x.view(3, -1)` 中的 `-1` 会被推算为 8。一次变形最多只能出现一个 `-1`。

```python
x.shape                      # (2, 3, 4)
x.transpose(1, 2).shape      # (2, 4, 3)
x.permute(1, 2, 0).shape     # (3, 4, 2)
```

`permute(1,2,0)` 表示新轴依次来自旧轴 1、2、0。本次 reshape 练习使用：

```python
z = x.view(2, 3, 4)
y = z.permute(1, 0, 2)
y = y.reshape(3, 8)
```

思考方式是“先分组 → 调整组顺序 → 合并相邻维度”。

## 11. 归约操作

归约把多个元素聚合成较少的元素，例如求和、均值和最小值。

| 写法 | 作用 |
| --- | --- |
| `x.sum()` | 所有元素求和，得到标量张量 |
| `x.sum(dim=d)` | 沿第 `d` 维求和并消除该维 |
| `x.mean(dim=d)` | 沿第 `d` 维求平均值 |
| `x.min()` | 整个张量的最小值 |
| `x.min(dim=d)` | 同时返回指定维度的最小值和索引 |
| `x.argmin(dim=d)` | 只返回最小值索引 |
| `x.max()` | 最大值 |
| `x.abs()` | 逐元素取绝对值，不是归约 |
| `x.sqrt()` | 逐元素开平方，不是归约 |

对于 `x.shape == (2,3)`：

```python
x.sum(dim=0).shape  # (3,)：维度 0 被消除
x.sum(dim=1).shape  # (2,)：维度 1 被消除
x.sum(dim=1, keepdim=True).shape  # (2, 1)
```

不要只背“按行/按列”，应记住：`dim=d` 表示聚合掉形状中的第 `d` 维。

指定 `dim` 的 `min` 返回两个张量：

```python
values, indices = x.min(dim=1)
```

- `values`：每组的最小值。
- `indices`：最小值在被归约维度上的位置。

每行最小元素置零：

```python
y = x.clone()
_, min_indices = x.min(dim=1)
rows = torch.arange(x.shape[0], device=x.device)
y[rows, min_indices] = 0
```

## 12. 广播（broadcasting）

广播让不同形状但兼容的张量进行逐元素运算。若 `x.shape == (4,3)`、`v.shape == (3,)`，则 `x + v` 会把 `v` 看成每行相同的 `(4,3)` 张量。

从最后一维向前比较，每一维必须满足：大小相同、其中一个为 1，或其中一个张量没有这一维（可在左侧补 1）。

| 形状 A | 形状 B | 是否兼容 | 结果形状 |
| --- | --- | --- | --- |
| `(4, 3)` | `(3,)` | 是 | `(4, 3)` |
| `(2, 1, 4)` | `(3, 4)` | 是 | `(2, 3, 4)` |
| `(4, 3)` | `(4,)` | 否 | 最后一维 3 与 4 不兼容 |

列标准化利用了广播：

```python
M = x.shape[0]
mu = x.sum(dim=0) / M
centered = x - mu
variance = (centered ** 2).sum(dim=0) / (M - 1)
sigma = variance.sqrt()
y = centered / sigma
```

若 `x.shape == (M,N)`，则 `mu`、`sigma` 的形状都是 `(N,)`，会分别作用于 `x` 的每一行。

## 13. 矩阵与批量矩阵运算

```python
x * y           # 对应位置逐元素相乘
x @ w           # 矩阵乘法
x.mm(w)         # 二维矩阵乘法
torch.mm(x, w)  # 与 x.mm(w) 等价
```

若 `(N,M) @ (M,P)`，结果形状为 `(N,P)`，中间的 `M` 必须一致。

批量矩阵乘法：

```python
x.shape             # (B, N, M)
y.shape             # (B, M, P)
z = torch.bmm(x, y)
z.shape             # (B, N, P)
```

循环版本等价于：

```python
products = []
for i in range(x.shape[0]):
    products.append(x[i] @ y[i])
z = torch.stack(products, dim=0)
```

`torch.stack` 会创建新维度。若每个 `product` 形状为 `(N,P)`，堆叠 B 个后得到 `(B,N,P)`。

## 14. CPU 与 GPU

| 写法 | 作用 |
| --- | --- |
| `torch.cuda.is_available()` | 检查 CUDA GPU 是否可用 |
| `x.cuda()` | 移到默认 CUDA GPU |
| `x.cpu()` | 移回 CPU |
| `x.to("cuda")` | 另一种移到 GPU 的写法 |
| `x.to(device)` | 移到变量指定的设备 |
| `torch.cuda.synchronize()` | 等待 GPU 异步计算完成，精确计时时使用 |

```python
x_gpu = x.cuda()
w_gpu = w.cuda()
y_gpu = x_gpu.mm(w_gpu)  # 在 GPU 上计算矩阵乘法
y = y_gpu.cpu()          # 将结果移回 CPU
```

同一次运算中的张量必须位于同一设备。通用设备选择：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
```

## 15. Colab 模块自动刷新与检查

Python 会缓存已经导入的模块。修改 `pytorch101.py` 后，可在 notebook 的设置区域启用：

```python
import importlib
import sys

sys.modules["imp"] = importlib
%load_ext autoreload
%autoreload 2
```

保存 `.py` 文件后，再运行测试单元格。检查实际导入路径和当前加载的函数源码：

```python
import inspect
import pytorch101

print(pytorch101.__file__)
print(inspect.getsource(pytorch101.mutate_tensor))
```

## 16. 常见错误速查

| 错误或现象 | 常见原因 | 处理方法 |
| --- | --- | --- |
| `NameError: name 'torch' is not defined` | 尚未执行 `import torch` | 先运行 Setup 和导入单元格 |
| `AttributeError: 'NoneType' ...` | 函数仍返回 `None` | 检查 TODO 区是否给返回变量赋值 |
| `IndexError: index ... is out of bounds` | 索引超过维度大小 | 索引从 0 开始；one-hot 列数用 `max(x)+1` |
| `SyntaxError` | 语法不完整，如 `[::2,1::2]` | 张量切片应写为 `x[::2,1::2]` |
| `IndentationError` | 混用 Tab 和空格或缩进层级不一致 | 统一使用 4 个空格 |
| 修改 `.py` 后结果不变 | 文件未保存、模块缓存或导入错路径 | 保存、启用 autoreload、检查 `__file__` |
| `view` 报内存布局错误 | 换轴后数据可能不连续 | 使用 `contiguous().view(...)` 或 `reshape(...)` |
| CPU/GPU device 不一致 | 运算中的张量不在同一设备 | 用 `.to(device)` 统一设备 |
| `arange` 范围错误 | 起止值与步长方向不一致 | 正步长时确保起点小于终点，并处理空范围 |

## 17. A1 函数与知识点对应表

| 作业函数 | 练习的主要知识 |
| --- | --- |
| `create_sample_tensor` | `torch.tensor`、嵌套列表、形状 |
| `mutate_tensor` | `zip`、索引赋值、原地修改 |
| `count_tensor_elements` | `x.shape`、循环、累乘 |
| `create_tensor_of_pi` | `torch.full` |
| `multiples_of_ten` | `arange`、包含边界、空张量、`float64` |
| `slice_indexing_practice` | 普通索引、切片、维度保留 |
| `slice_assignment_practice` | 切片赋值、限制修改范围 |
| `shuffle_cols` / `reverse_rows` | 整数张量高级索引 |
| `take_one_elem_per_col` | 成对的行列索引 |
| `make_one_hot` | `zeros`、`arange`、高级索引赋值 |
| `sum_positive_entries` | 布尔掩码、`sum`、`item` |
| `reshape_practice` | `view`、`permute`、`reshape` |
| `zero_row_min` | `clone`、`min(dim=...)`、高级索引 |
| `batched_matrix_multiply_*` | `@`、`stack`、`torch.bmm` |
| `normalize_columns` | 归约、广播、样本标准差 |
| `mm_on_cpu` / `mm_on_gpu` | 设备移动、矩阵乘法、GPU 加速 |

## 18. 最值得优先记住的写法

```python
x = torch.tensor(data, dtype=torch.float32)
x.shape
x[i, j]
x[:, 2:3]
y = x.clone()
y[rows, cols] = values
x.reshape(new_shape)
x.permute(new_axis_order)
x.sum(dim=d)
values, indices = x.min(dim=d)
x[x > 0]
x @ w
torch.bmm(x, y)
x = x.to(device)
```

遇到不确定的题目时，先写出每一步输入和输出的 `shape`，再决定索引、归约或形状变换应该作用在哪个 `dim`。
