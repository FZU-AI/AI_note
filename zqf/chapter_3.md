# 线性回归
## ⽮量计算表达式
先考虑对两个向量相加的两种⽅法
```python
from mxnet import nd
from time import time
a = nd.ones(shape=1000)
b = nd.ones(shape=1000)
```
第一种，按元素逐⼀做标量加法。
```python
In [2]: start = time()
c = nd.zeros(shape=1000)
for i in range(1000):
c[i] = a[i] + b[i]
time() - start
Out[2]: 0.15223002433776855
```
第二种，将这两个向量直接做⽮量加法。

```python
In [3]: start = time()
d = a + b
time() - start
Out[3]: 0.00029015541076660156
```
**结论，两个向量直接做⽮量加法更省时。我们应该尽可能采⽤⽮量计算，以提升计算效率。**
