# KNN 高频错误与排查清单

这部分比“记住正确答案”更重要。我们在 KNN 练习里反复遇到的很多问题并不是算法错误，而是缩进、Notebook 状态、模块缓存和张量方向错误。

## 1. 最高频错误：缩进

Python 使用缩进表示代码块。缩进不仅影响能否运行，也会直接改变算法逻辑。

### 标准层级

```python
def function(...):                 # 0 层
    for k in k_choices:            # 4 个空格
        for i in range(num_folds): # 8 个空格
            value = ...            # 12 个空格
```

同一层必须严格对齐，并统一使用 4 个空格，不要混用 Tab 和空格。

### 错误 A：循环后没有缩进

```python
for j in range(num_test):
sorted_indices = torch.argsort(dists[:, j])
```

这会产生：

```text
IndentationError: expected an indented block after 'for' statement
```

由于 `knn.py` 整个模块无法解析，甚至 `from knn import hello` 也会失败。报错虽然出现在导入时，根因却可能位于文件后面尚未完成的函数。

### 错误 B：能运行，但循环体范围错了

```python
for j in range(num_test):
    nearest_labels = ...

label_counts = torch.bincount(nearest_labels)
y_pred[j] = label_counts.argmax()
```

这段代码没有语法错误，但投票和赋值已经退出循环，只会处理最后一个测试样本。其他预测仍然保持初始化值 `0`，测试通常显示 `Correct: False`。

正确逻辑必须让“排序、取邻居、投票、写预测”全部位于循环内部。

### 错误 C：交叉验证的 `append` 放错层级

```python
for k in k_choices:
    for i in range(num_folds):
        accuracy = ...

    k_to_accuracies[k].append(accuracy)
```

这样每个 `k` 只保存最后一折的结果。正确位置是在内层 `for i` 中，使每个 `k` 得到 `num_folds` 个准确率。

### 快速自检

- 函数体：4 个空格；
- 第一层循环：8 个空格；
- 第二层循环：12 个空格；
- 选中代码后用 Tab / Shift+Tab 整体调整；
- 编辑器中开启“显示空白字符”；
- 出现缩进错误时，删除问题行行首空白并重新按空格键输入。

## 2. 保存了 `.py`，不代表 Notebook 已在使用新代码

Jupyter/Colab 是有状态环境。代码、导入的模块和输出可能分别来自不同时间。

典型现象：

- 右侧 `knn.py` 已改正确，左侧仍输出旧结果；
- `predict` 已经写好，但 Notebook 仍然得到 `None`；
- 修改后没有重新运行测试单元格，旧的 `Correct: False` 仍显示在页面上；
- `autoreload` 自己报错，于是模块没有刷新。

推荐顺序：

1. 保存 `knn.py`；
2. 确认文件没有语法/缩进错误；
3. 重新执行导入和测试单元格；
4. 若状态混乱，重启运行时并从上到下重新运行。

可检查实际导入的文件：

```python
import knn
print(knn.__file__)
```

也可临时检查运行时中的函数源码：

```python
import inspect
print(inspect.getsource(knn.predict_labels))
```

检查完成后应删除临时调试单元格，保持作业 Notebook 干净。

### Python 3.13 的 `autoreload` 兼容问题

旧版课程环境可能依赖已经移除的 `imp` 模块。在原 autoreload 单元格中可使用：

```python
import importlib
import sys

sys.modules["imp"] = importlib

%load_ext autoreload
%autoreload 2
```

即使启用了 autoreload，文件存在语法错误时依然无法重新载入。

## 3. `NoneType` 往往是上游函数没有返回结果

错误示例：

```text
TypeError: 'NoneType' object is not iterable
```

它曾出现在：

```python
test_colors = [class_colors[c] for c in y_test]
```

这通常不是绘图代码的问题，而是 `classifier.predict(...)` 返回了 `None`。应沿数据流向上检查：

```text
predict
  -> compute_distances_no_loops
  -> predict_labels
  -> return y_test_pred
```

尤其检查：是否仍有 `pass`、是否真的给返回变量赋值、`return` 的变量名是否一致。

## 4. 空白交叉验证图不是绘图问题

如果坐标轴显示出来但没有数据点，先检查：

```python
print(k_to_accuracies)
```

正常结果应类似：

```python
{1: [27.0, 26.0, 28.0, 25.0, 27.5],
 3: [28.0, 27.0, 29.0, 26.5, 28.5]}
```

若结果是 `{}`，说明尚未运行交叉验证单元格、运行顺序错误，或者函数提前返回空字典。绘图单元格只负责展示已有结果，不会替你执行交叉验证。

## 5. `...` 不是“以后补上”的注释

下面的代码不是完整实现：

```python
x_train_cv = torch.cat(...)
```

`...` 在 Python 中是真实的 `Ellipsis` 对象。它可以通过语法检查，但传给 `torch.cat` 会在运行时失败。必须换成真正的张量列表：

```python
x_train_folds[:i] + x_train_folds[i + 1:]
```

## 6. 距离矩阵的方向必须始终一致

本作业规定：

```text
dists.shape == (num_train, num_test)
dists[i, j] == train i 与 test j 的距离
```

因此预测第 `j` 个测试样本时使用：

```python
dists[:, j]
```

不是 `dists[j, :]`。调试时先打印：

```python
print(dists.shape)
print(num_train, num_test)
```

## 7. 分清“值”和“位置”

- `torch.sort` / `torch.topk` 通常同时返回值和索引；
- `torch.argsort` 只返回排序后的索引；
- `torch.min(x, dim=...)` 返回 `(values, indices)`；
- `torch.argmin` / `torch.argmax` 只返回位置。

KNN 要用的是最近训练样本的**位置**，然后用这些位置索引 `y_train`。不要把距离值误当成标签下标。

## 8. `topk` 的方向别写反

`torch.topk` 默认取最大值，而最近邻要取最小距离：

```python
_, indices = torch.topk(distances, k, largest=False)
```

漏写 `largest=False` 会找到最远邻居，代码仍能运行，却得到错误预测。

## 9. `torch.bincount` 和 `argmax`

```python
neighbor_labels = torch.tensor([1, 0, 1])
votes = torch.bincount(neighbor_labels)  # tensor([1, 2])
prediction = votes.argmax()              # tensor(1)
```

`bincount` 的输入必须是非负整数张量。`argmax` 返回票数最大元素的位置，该位置正好就是类别编号。

## 10. 每完成一层就检查形状和最小测试

不要等完整 CIFAR-10 实验跑几分钟后才发现基础错误。推荐顺序：

1. `python -m py_compile A1/knn.py`：先排除语法和缩进错误；
2. 用小张量验证 `dists.shape`；
3. 比较双循环、单循环、无循环三个结果是否接近；
4. 用课程给出的 `predict_labels` 小例子检查 `[1, 0, 0]`；
5. 检查每个 `k` 是否恰好有 `num_folds` 个准确率；
6. 最后再运行大数据集。

### 最终提交前清单

- [ ] 所有 TODO 中不再残留必做的 `pass` 或 `...`
- [ ] `knn.py` 已保存，且能正常导入
- [ ] 三种距离计算的形状均为 `(num_train, num_test)`
- [ ] `predict_labels` 对每个测试样本都完成投票和赋值
- [ ] `KnnClassifier.predict` 返回张量而不是 `None`
- [ ] 每个 `k` 保存了 `num_folds` 个交叉验证结果
- [ ] 最佳 `k` 的并列情况选择更小值
- [ ] Notebook 按顺序重新运行，输出不是旧缓存
- [ ] 删除临时调试单元格和最后多余的空单元格
