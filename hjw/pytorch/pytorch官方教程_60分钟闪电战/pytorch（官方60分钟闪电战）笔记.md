# 安装

anaconda3安装<https://blog.csdn.net/ITLearnHall/article/details/81708148>

pytorch安装<https://blog.csdn.net/thomaswu1992/article/details/90293015>

官方60分钟闪电战链接：<https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py>

安装后：导入模块不报错即为安装成功

```python
import torch
import torchvison
```

---

# Tensors（张量）

Tensors 类似于 NumPy 的 ndarrays ，同时 Tensors 可以使用 GPU 进行计算 

## **tensor的创建**

### 1.直接创建tensor

> torch.tensor( data , dtype=None , device=None , requires_grad=False )
>
> data - 可以是list, tuple, numpy array, scalar或其他类型
>
> dtype - 可以返回想要的tensor类型
>
> device - 可以指定返回的设备
>
> requires_grad - 可以指定是否进行记录图的操作，默认为False
>
> 需要注意的是，torch.tensor 总是会复制 data, 如果你想避免复制，可以使 torch.Tensor. detach()，如果是从 numpy 中获得数据，那么你可以用 torch.from_numpy(), 注from_numpy() 是共享内存的

---

### 2.根据数值要求来创建

> ```python
> torch.zeros(*sizes, out=None, ..)# 返回大小为sizes的零矩阵 
> 
> torch.zeros_like(input, ..) # 返回与input相同size的零矩阵
> 
> torch.ones(*sizes, out=None, ..) #f返回大小为sizes的单位矩阵
> 
> torch.ones_like(input, ..) #返回与input相同size的单位矩阵
> 
> torch.full(size, fill_value, …) #返回大小为sizes,单位值为fill_value的矩阵
> 
> torch.full_like(input, fill_value, …) 返回与input相同size，单位值为fill_value的矩阵
> 
> torch.arange(start=0, end, step=1, …) #返回从start到end, 单位步长为step的1-d tensor.
> 
> torch.linspace(start, end, steps=100, …)  #返回从start到end, 间隔中的插值数目为steps											   的1-d tensor
> 
> torch.logspace(start, end, steps=100, …) #返回1-d tensor ，从10^start到10^end的											   steps个对数间隔
> ```

### 3.按照矩阵要求

> ```python
> torch.eye(n, m=None, out=None,…) #返回2-D 的单位对角矩阵
> 
> torch.empty(*sizes, out=None, …) #返回被未初始化的数值填充，大小为sizes的tensor
> 
> torch.empty_like(input, …) # 返回与input相同size,并被未初始化的数值填充的tensor
> ```

### 4.随机生成矩阵

> ```python
> torch.normal(mean, std, out=None)
> 
> torch.rand(*size, out=None, dtype=None, …) #返回[0,1]之间均匀分布的随机数值
> 
> torch.rand_like(input, dtype=None, …) #返回与input相同size的tensor, 填充均匀分布的随机数值
> 
> torch.randint(low=0, high, size,…) #返回均匀分布的[low,high]之间的整数随机值
> 
> torch.randint_like(input, low=0, high, dtype=None, …) #
> 
> torch.randn(*sizes, out=None, …) #返回大小为size,由均值为0，方差为1的正态分布的随机数值
> 
> torch.randn_like(input, dtype=None, …)
> 
> torch.randperm(n, out=None, dtype=torch.int64) # 返回0到n-1的数列的随机排列
> ```

---

## tensor的操作

**加法运算**

```python
import torch
# 创建两个初始化随机的tensor
x = torch.randn(2,2)
y = torch.randn(2,2)

#通过运算符实现加法
print(x+y)

#通过torch.add()
print(torch.add(x,y))

#提供输出 Tensor 作为参数
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

#通过tensor张量自身实现
print(y.add_(x))
```

---

**索引操作**

任何使张量会发生变化的操作都有一个前缀 ‘’。例如：x.copy(y), x.t_(), 将会改变 x 

```python
print(x[:, 1])
```

其索引跟numpy的索引操作类似

---

**改变大小**

如果你想改变一个 tensor 的大小或者形状，你可以使用 torch.view: 

```python
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
```

**输出值**

如果你有一个元素 tensor ，使用 .item() 来获得这个 value 。 

```python
x = torch.randn(1)
print(x)
print(x.item())
```

注意：.item() 只能得到一个值

```python
# 通过.item来获得tensor的value
x = torch.tensor([2,3])
i = x[1].item()            #使用索引
i						# i 是int类型的数据
```

---

**Numpy转换**

Converting a Torch Tensor to a NumPy array and vice versa is a breeze.

将 PyTorch 张量转换为 NumPy 数组（反之亦然）是一件轻而易举的事

The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.

**PyTorch 张量和 NumPy 数组将共享其底层内存位置，改变一个也将改变另一个。**

---

将 PyTorch 张量转换为 NumPy 数组：

```python
# 将 PyTorch 张量转换为 NumPy 数组：
a = torch.ones(5)
b = a.numpy()
print(a,b)
```

观察NumPy 数组的值如何变化：

```python
a.add_(1)
print(a,b)
```

**a,b都同时变化了**

NumPy 数组转换成 PyTorch 张量时，可以使用 `from_numpy` 完成

```python
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
a, b
```

输出结果

```
(array([2., 2., 2., 2., 2.]),
 tensor([2., 2., 2., 2., 2.], dtype=torch.float64))
```

同样的，a只要发送改变，b就会相应的发生改变

---

### Autograd自动求导

- PyTorch 中所有神经网络的核心是 [`autograd`](https://pytorch.org/docs/stable/autograd.html?highlight=autograd#module-torch.autograd)

- autograd为张量上的所有操作提供自动求导的功能

- 设置属性.requires_grad为true,会追踪所有对该张量的操作当完成计算后通过调用 `.backward()` 会自动计算所有的梯度，这个张量的所有梯度将会自动积累到 `.grad` 属性。这也就完成了自动求导的过程

阻止张量追踪历史记录：

- 可以调用 `.detach()` 方法将其与计算历史记录分离
- 将代码块包装在 `with torch.no_grad():` 语句中

---

# 神经网络

PyTorch 中，我们可以使用 [`torch.nn`](https://pytorch.org/docs/stable/nn.html?highlight=torch%20nn#module-torch.nn)来构建神经网络。

神经网络的典型训练过程如下：

1. 定义包含可学习参数（权重）的神经网络模型。
2. 在数据集上迭代。
3. 通过神经网络处理输入。
4. 计算损失（输出结果和正确值的差值大小）。
5. 将梯度反向传播回网络节点。
6. 更新网络的参数，一般可使用梯度下降等最优化方法。

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
net
```

输出结果:

> ```
> Net(
>   (conv1): Conv2d(1, 6, kernel_size=(3, 3), stride=(1, 1))
>   (conv2): Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1))
>   (fc1): Linear(in_features=576, out_features=120, bias=True)
>   (fc2): Linear(in_features=120, out_features=84, bias=True)
>   (fc3): Linear(in_features=84, out_features=10, bias=True)
> )
> ```

---

通过net.parameters()获取模型参数

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

---

整个网络流程：

```
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
```

```python
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU
```

next_functions是倒推的一个过程

---

通过loss.backward（）完成反向传播，**在反向传播前，需求清空现存的梯度，否则梯度会累加在一起**

**更新权重**

weight = weight - learning_rate * gradient

```python
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
```

---

# 图像分类器

训练图像分类器步骤：

1.使用torchvision加载并且归一化CIFAR10的训练和测试数据集

2.定义一个卷积神经网络

3.定义一个损失函数

4.通过训练集训练网络

5.通过测试集测试网络