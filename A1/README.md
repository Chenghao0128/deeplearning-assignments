# Assignment 1

本目录记录 Assignment 1 的代码、课程 notebook 和学习笔记。

## PyTorch 101

状态：主线练习已完成；文件末尾两个 `challenge_*` 函数属于 notebook 未要求的附加挑战，目前保留原始 `pass`。

主要文件：

- `pytorch101.py`：作业实现。
- `pytorch101.ipynb`：课程讲义及测试。
- `notes/pytorch101-notes.md`：中文学习记录。

## 在 Google Colab 中运行

1. 将 `A1` 文件上传到 Google Drive。
2. 在 Colab 中打开 `pytorch101.ipynb`。
3. 挂载 Google Drive，并将 notebook 中的路径设置为实际的 `A1` 路径。
4. 按顺序运行设置和测试单元格。
5. GPU 练习需在“代码执行程序 → 更改运行时类型”中选择 GPU。

### 当前 Colab 的 autoreload 兼容处理

课程 notebook 使用旧版 IPython 的 `autoreload`，在移除了 `imp` 的新版 Python 中可能报错。可在原 autoreload 单元格使用：

```python
import importlib
import sys

sys.modules["imp"] = importlib

%load_ext autoreload
%autoreload 2
```

这样，保存 `pytorch101.py` 后，下一次运行 notebook 单元格时会自动载入最新代码。

## 完成内容

- 张量创建、修改、形状与元素数量
- 数据类型与张量构造函数
- 切片、整数数组索引和布尔索引
- one-hot 编码
- 张量形状变换与轴交换
- 归约、广播和列标准化
- 普通与批量矩阵乘法
- CPU / CUDA 设备切换与 GPU 加速

