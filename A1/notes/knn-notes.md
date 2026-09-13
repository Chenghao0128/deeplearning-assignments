# K-Nearest Neighbors 学习笔记

这份笔记整理 Assignment 1 中 KNN 分类器的完整主线：距离计算、标签预测、分类器封装、交叉验证以及最佳 `k` 的选择。

## 1. KNN 在做什么

KNN（K-Nearest Neighbors，K 近邻）几乎没有传统意义上的“训练”。它会直接保存训练数据；预测一个测试样本时：

1. 计算该测试样本与所有训练样本的距离；
2. 找到距离最小的 `k` 个训练样本；
3. 统计这 `k` 个邻居的标签；
4. 将票数最多的标签作为预测结果。

因此，KNN 的训练很快，但预测需要和大量训练样本比较，速度和内存开销都较大。

## 2. 先看清张量形状

课程中的图像张量通常具有以下形状：

```text
x_train: (num_train, C, H, W)
x_test:  (num_test,  C, H, W)
dists:   (num_train, num_test)
```

最重要的约定是：

```text
dists[i, j] = 第 i 个训练样本与第 j 个测试样本之间的平方欧氏距离
```

所以固定 `j` 后，`dists[:, j]` 才是“第 `j` 个测试样本到全部训练样本的距离”。行列方向一旦弄反，后面的近邻索引和预测都会错。

## 3. 平方欧氏距离

两个样本 `a` 和 `b` 的平方欧氏距离为：

```text
||a - b||² = Σ(a - b)²
```

课程要求的是平方距离，不需要再开平方。平方根是单调函数，不会改变距离从小到大的顺序，因此不会影响最近邻选择。

### 双循环版本

最直观的实现逐个选择训练样本和测试样本：

```python
for i in range(num_train):
    for j in range(num_test):
        difference = x_train[i] - x_test[j]
        dists[i, j] = (difference ** 2).sum()
```

它容易理解，但 Python 循环会成为性能瓶颈。

### 单循环与广播

固定一个训练样本 `x_train[i]` 后：

```text
x_train[i]: (C, H, W)
x_test:     (num_test, C, H, W)
```

表达式 `x_train[i] - x_test` 会通过广播，同时得到它与所有测试样本之差。再把每个测试样本展平，对特征维求和：

```python
difference = x_train[i] - x_test
dists[i] = (difference ** 2).reshape(num_test, -1).sum(dim=1)
```

### 无循环向量化版本

使用恒等式：

```text
||a - b||² = ||a||² + ||b||² - 2aᵀb
```

先将图像展平成特征向量：

```text
train: (num_train, D)
test:  (num_test, D)
```

各部分形状如下：

| 表达式 | 形状 | 含义 |
|---|---:|---|
| `(train ** 2).sum(dim=1, keepdim=True)` | `(num_train, 1)` | 每个训练样本的平方和 |
| `(test ** 2).sum(dim=1)` | `(num_test,)` | 每个测试样本的平方和 |
| `train @ test.t()` | `(num_train, num_test)` | 所有训练与测试样本的内积 |

利用广播把三部分组合起来：

```python
dists = train_squared + test_squared - 2 * (train @ test.t())
```

这里最值得掌握的思路不是背代码，而是每写一个表达式就先推导它的形状。

## 4. 根据距离预测标签

对每个测试样本 `j`：

```python
sorted_indices = torch.argsort(dists[:, j])
nearest_indices = sorted_indices[:k]
nearest_labels = y_train[nearest_indices]
label_counts = torch.bincount(nearest_labels)
y_pred[j] = label_counts.argmax()
```

各函数的职责：

- `torch.argsort(...)` 返回从小到大排序后的**位置索引**；
- `[:k]` 选择距离最小的 `k` 个位置；
- `y_train[nearest_indices]` 用高级索引取得邻居标签；
- `torch.bincount(...)` 统计每个非负整数标签出现的次数；
- `.argmax()` 返回票数最多的标签位置。

如果多个标签票数相同，`argmax()` 会返回最先出现的位置，也就是较小的标签，符合课程要求。

也可以用 `torch.topk(dists[:, j], k, largest=False)` 直接取得最小的 `k` 个距离及其索引。这里的关键是 `largest=False`。

## 5. `KnnClassifier` 类

初始化方法只保存数据，不进行计算：

```python
self.x_train = x_train
self.y_train = y_train
```

`predict` 方法把已经实现的两个步骤连接起来：

```python
dists = compute_distances_no_loops(self.x_train, x_test)
y_test_pred = predict_labels(dists, self.y_train, k=k)
```

`self` 表示当前分类器对象。使用 `self.x_train` 后，预测时就不需要再次传入训练数据。

`check_accuracy` 的计算为：

```python
num_correct = (y_test == y_test_pred).sum().item()
accuracy = 100.0 * num_correct / num_samples
```

其中 `.item()` 将单元素张量转换为 Python 数值。

## 6. `k` 对预测边界的影响

- `k=1`：只听最近的一个样本，决策边界细碎，对噪声敏感，容易过拟合；
- 中等的 `k`：综合多个邻居，边界更平滑，通常泛化更好；
- `k` 太大：远处、不相关的样本也参与投票，容易欠拟合。

我们在 5000 个训练样本和 500 个测试样本的实验中观察到：

- `k=1`：约 `27.4%`；
- `k=5`：约 `27.8%`。

这说明改变 `k` 会改变结果，但不能仅凭一次测试集成绩选择超参数。

## 7. 交叉验证

交叉验证用于选择 `k`，测试集只应在最后评估一次。

以 5 折交叉验证为例：

1. 用 `torch.chunk` 把训练数据分成 5 折；
2. 每次取一折作为验证集；
3. 用 `torch.cat` 拼接其余 4 折作为训练集；
4. 对每个候选 `k` 重复 5 次；
5. 将准确率保存在 `k_to_accuracies[k]` 中。

核心结构是：

```python
for k in k_choices:
    k_to_accuracies[k] = []
    for i in range(num_folds):
        x_val = x_train_folds[i]
        y_val = y_train_folds[i]
        x_train_cv = torch.cat(x_train_folds[:i] + x_train_folds[i + 1:], dim=0)
        y_train_cv = torch.cat(y_train_folds[:i] + y_train_folds[i + 1:], dim=0)
        classifier = KnnClassifier(x_train_cv, y_train_cv)
        accuracy = classifier.check_accuracy(x_val, y_val, k=k, quiet=True)
        k_to_accuracies[k].append(accuracy)
```

每个 `k` 对应的列表长度必须等于 `num_folds`。最佳 `k` 是平均验证准确率最高者；如果平均值并列，则选择更小的 `k`。

完整数据集包含 50000 个训练样本和 10000 个测试样本，计算会明显更久。课程给出的目标是最终准确率超过 33%；仓库代码已具备运行完整实验的条件，但本地整理阶段没有虚构一次未执行的完整数据集结果。

## 8. KNN 的局限

- 图像像素的欧氏距离不一定代表语义相似度；
- 亮度、平移和背景变化都可能显著改变像素距离；
- 预测需要比较大量训练样本，时间复杂度高；
- 需要保存完整训练集，内存开销大。

尽管如此，KNN 很适合作为基线方法，也很好地串联了张量形状、广播、矩阵乘法、高级索引、归约和交叉验证等基础知识。
