# Stride

> 参考资料
> - [NumPy ndarray.strides](https://numpy.org/devdocs/reference/generated/numpy.ndarray.strides.html?utm_source=chatgpt.com)
>
> 阅读后文前建议阅读以上参考资料

我们运行一段 NumPy 代码：
``` Python
>>> import numpy as np
>>> y = np.reshape(np.arange(2 * 3 * 4, dtype=np.int32), (2, 3, 4))
>>> y
array([[[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]],

       [[12, 13, 14, 15],
        [16, 17, 18, 19],
        [20, 21, 22, 23]]], dtype=int32)
>>> y.strides
(48, 16, 4)
>>> y[1, 1, 1]
np.int32(17)
>>> offset = sum(y.strides * np.array((1, 1, 1)))
>>> offset // y.itemsize
np.int64(17)
```

代码中出现了 `strides` 的概念。在 NumPy 中，`strides` 表示沿着每个维度前进一步时，需要跨过的字节数。

在示例代码中，`y` 是一个形状为 `(2, 3, 4)` 的数组，数组中的每个数占用 `32 bits = 4 bytes`。因此，在默认采用 [row-major](/docs/Chapter1/Chapter1_1.md#row-major行优先) 的情况下：
- 第三个维度（`shape[2] = 4`）每前进一步需要经过 4 bytes，因为 index 每增加 1，地址就增加 4 bytes
- 第二个维度（`shape[1] = 3`）每前进一步需要经过 16 bytes，因为需要跨过第三个维度的 4 个元素，bytes=$4\times 4=16$
- 第一个维度（`shape[0] = 2`）每前进一步需要经过 48 bytes，bytes=$4\times 3\times 4=48$

因此，在上述示例中，输入 `y.strides` 得到的输出是 `(48, 16, 4)`。

但是在 PyTorch 这样的深度学习框架中，stride 通常以元素为单位。因此，`y.stride()` 会得到 `(12, 4, 1)`。在后面的讲解中，我们同样以元素为单位进行描述。

## 相同形状，不同一维排列
Stride 计算元素位置时需要结合 shape。总的来说，公式类似前文的 `offset = sum(y.strides * np.array((i, j, k)))`，这里的 i、j、k 是各个维度上的索引。形式化地说，我们可以有以下公式：

$$
offset(i_{0},i_{1},\dots,i_{n-1})=\sum_{k=0}^{n-1}i_{k}\cdot strides[k],\quad n=shape.size()
$$

我们举一个非 row-major 的例子，假设一个 `shape = [2, 3]` 的矩阵：
$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}
$$

在传统的 row-major 下，也就是 `stride = [3, 1]` 时，它的一维排列是 `[1, 2, 3, 4, 5, 6]`。

但是当我们将 stride 更改为 `[1, 3]` 时，根据上面的公式，它的一维排列则变成 `[1, 4, x, 2, 5, x, 3, 6]`，其中 x 代表空隙。

我们会发现，即使形状相同，stride 不同，索引映射也会不同。

## 反过来用呢？
对于一维排列 `[1, 2, 3, 4, 5, 6]`，我们假定它的 `shape = [3, 2]`、`strides = [1, 3]`，那么我们会得到：
$$
\begin{bmatrix}
1 & 4 \\
2 & 5 \\
3 & 6
\end{bmatrix}
$$

没错，这正是原矩阵的转置。这意味着，我们没有修改数据排布，仅仅通过改变 shape 和 strides，就实现了数据的转置。
