# PyTorch 101 学习记录

## 1. 张量基础

张量（Tensor）是 PyTorch 的核心数据结构。二维张量可看作矩阵，形状 `(M, N)` 表示 `M` 行、`N` 列。索引从 `0` 开始。

```python
x = torch.tensor([[0, 10], [100, 0], [0, 0]])
x.shape  # torch.Size([3, 2])
```

嵌套列表的每个内层列表表示一行。`torch.tensor()` 接收整个嵌套列表作为一个参数，不能把每一行作为独立位置参数传入。

常用构造函数：

```python
torch.zeros(2, 3)              # 2×3 全零张量
torch.ones(1, 2)               # 1×2 全一张量
torch.full((2, 3), 3.14)       # 指定形状和填充值
torch.eye(3)                   # 3×3 单位矩阵
torch.rand(4, 5)               # [0, 1) 均匀随机数
torch.arange(10, 50, 10)       # 10, 20, 30, 40；末端不包含
torch.empty((0,))              # 形状为 (0,) 的空张量
```

## 2. Python 循环、zip 与原地修改

`zip(a, b)` 按位置配对两个可迭代对象。下面的代码把坐标与新值配对，并直接修改输入张量：

```python
for index, value in zip(indices, values):
    x[index] = value
```

如果同一索引重复出现，后一次赋值会覆盖前一次。直接修改传入对象称为原地修改（in-place mutation）。

`*=` 是乘后赋值的简写：

```python
count *= dimension
# 等价于 count = count * dimension
```

张量元素总数等于所有形状维度的乘积。

## 3. 数据类型 dtype

每个张量都有 `dtype`。常用类型：

- `torch.float32`：深度学习参数、输入和激活最常用的浮点类型。
- `torch.float64`：更高精度的浮点类型。
- `torch.int64`：常用于索引和类别编号。
- `torch.bool`：布尔掩码。
- `torch.float16`：常用于 GPU 混合精度训练。

```python
torch.tensor([1, 2], dtype=torch.float64)
x.float()                 # float32
x.double()                # float64
x.to(torch.float32)
torch.zeros_like(x)       # 与 x 相同形状和 dtype
```

## 4. 切片索引

基础语法是 `x[start:stop:step]` 和 `x[row_slice, column_slice]`。`stop` 不包含在结果中；负索引从末尾计数。

```python
x[-1, :]      # 最后一行，一维结果 (N,)
x[:, 2:3]     # 第三列，保持二维 (M, 1)
x[:2, :3]     # 前两行、前三列
x[::2, 1::2]  # 偶数索引行、奇数索引列
```

整数索引会消除对应维度，而长度为 1 的切片会保留维度：

```python
x[:, 2].shape    # (M,)
x[:, 2:3].shape  # (M, 1)
```

切片通常返回共享原数据的视图。需要独立副本时使用 `y = x.clone()`。

## 5. 整数数组索引与 one-hot

两个索引数组会逐项配对：

```python
rows = [1, 0, 3]
cols = [0, 1, 2]
y = x[rows, cols]
# 得到 [x[1, 0], x[0, 1], x[3, 2]]
```

one-hot 编码：

```python
N = len(x)
C = max(x) + 1
y = torch.zeros((N, C), dtype=torch.float32)
rows = torch.arange(N)
y[rows, x] = 1
```

列数必须是 `max(x) + 1`，因为索引从 `0` 开始。

## 6. 布尔索引

```python
mask = x > 0
positive_values = x[mask]
positive_sum = positive_values.sum().item()
```

比较操作产生布尔掩码；`.item()` 把只包含一个值的 PyTorch 标量转换成 Python 标量。

## 7. 形状变换

`view()` 和 `reshape()` 改变形状，但不改变元素总数：

```python
x.view(-1)     # 展平为一维
x.view(1, -1)  # 一行二维张量
x.view(-1, 1)  # 一列二维张量
```

`-1` 表示该维度由 PyTorch 自动计算。交换轴要使用：

```python
x.t()                   # 二维转置
x.transpose(1, 2)       # 交换两个轴
x.permute(1, 0, 2)      # 指定所有轴的新顺序
```

复杂变形可以先为每个轴命名：

```text
(24,) → (group=2, row=3, col=4)
      → permute 为 (row=3, group=2, col=4)
      → 合并为 (3, 8)
```

`transpose` / `permute` 后张量可能不连续。可使用 `x.contiguous().view(...)`，或用更灵活的 `x.reshape(...)`。

## 8. 归约操作

归约把多个元素聚合成更少的元素，例如 `sum`、`mean`、`min` 和 `max`。

```python
x.sum()       # 所有元素求和
x.sum(dim=0)  # 消除 dim 0
x.sum(dim=1)  # 消除 dim 1
```

最可靠的记忆方式是：`dim=d` 会归约第 `d` 个维度。设置 `keepdim=True` 时，该维度会保留为长度 `1`。

```python
values, indices = x.min(dim=1)
indices_only = x.argmin(dim=1)
```

`min` 返回值和索引，`argmin` 只返回最小值的位置。

## 9. 广播

广播让形状不同但兼容的张量执行逐元素运算。比较形状时从右向左：维度长度相同，或其中一个为 `1`，即为兼容。

```text
(4, 3)
   (3,) → 补成 (1, 3)
结果形状为 (4, 3)
```

例如 `(4, 3)` 的矩阵加 `(3,)` 的向量，相当于每一行都加同一个向量。通常只是“表现得像复制”，并不会真的复制全部数据。

## 10. 列标准化

```python
M = x.shape[0]
mu = x.sum(dim=0) / M
centered = x - mu
variance = (centered ** 2).sum(dim=0) / (M - 1)
sigma = variance.sqrt()
y = centered / sigma
```

`mu` 和 `sigma` 的形状都是 `(N,)`，可通过广播作用到 `(M, N)` 的每一行。

## 11. 矩阵乘法与批量矩阵乘法

```text
(A, B) @ (B, C) → (A, C)
```

等价写法包括 `x.mm(w)`、`torch.mm(x, w)` 和 `x @ w`；`*` 是逐元素乘法。

批量矩阵乘法：

```text
(B, N, M) 与 (B, M, P) → (B, N, P)
```

循环版本逐批执行 `x[i] @ y[i]`，再用 `torch.stack(..., dim=0)` 堆叠；无循环版本使用 `torch.bmm(x, y)`。

## 12. CPU、GPU 与 CUDA

```python
x.device
torch.cuda.is_available()
x_gpu = x.cuda()
x_gpu = x.to("cuda")
x_cpu = x_gpu.cpu()
```

同一次运算中的张量必须位于同一设备。GPU 矩阵乘法流程：

```text
CPU 输入 → .cuda() → GPU 计算 → .cpu() → CPU 输出
```

测试中的 `torch.cuda.synchronize()` 等待异步 GPU 运算完成，从而正确计时。本次练习中 GPU 相比 CPU 获得了约 `5.98×` 加速。

## 13. 本次遇到的问题

### 修改 `.py` 后 notebook 仍运行旧代码

Python 会缓存已导入模块。课程使用的旧版 `autoreload` 在新版 Python 中依赖已经移除的 `imp`，导致自动重载失败。兼容方法见 `A1/README.md`。

### 缩进错误

复制代码时混用 Tab、普通空格或特殊空格，会产生 `IndentationError`。解决方法是删除行首空白，用编辑器重新缩进；函数体通常为 4 个空格，循环或条件内部为 8 个空格。

### 文件已保存不等于内存函数已更新

保存只更新磁盘文件；重新导入或成功启用 `autoreload` 才会更新运行时中的函数。

### 可见测试不一定覆盖通用输入

例如测试输入本来就是零时，遗漏“主动将某区域写为零”的代码仍可能通过；固定使用四行索引也可能通过固定形状测试。实现应遵循函数说明，而不仅是拟合示例。

## 14. Git 基础工作流

```bash
git status                 # 查看哪些文件发生变化
git add <path>             # 选择本次提交的文件
git commit -m "message"   # 保存一个有说明的版本
git push                   # 上传到 GitHub
git pull                   # 获取 GitHub 上的新版本
```

推荐每完成一个独立阶段就提交一次，例如：

```text
Complete A1 PyTorch 101 exercises and notes
```

