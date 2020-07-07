# 第三章 深度学习基础

#### 线性回归

**基础知识**

![](https://img2018.cnblogs.com/blog/1813159/202002/1813159-20200214151129329-1552445965.png)

**实现过程**

![](https://img2018.cnblogs.com/blog/1813159/202002/1813159-20200214151321110-1768674041.png)

**学习笔记**

**批量读取**

```text
torch_data = Data.TensorDataset(features, labels)
dataset = Data.DataLoader(torch_data, batch_size, shuffle=True)
```

**定义模型的两种常见写法** 这两种方法是我比较喜欢的方法。 其中有两点需要注意：

1. 虽说他们在定义时，输入和输出的神经元个数是一样的，但`print(net)`结果是不同的，法二有Sequential外层。
2. 由于第一点的原因，这也导致了在初始化参数时，`net[0].weight`应改为`net.linear.weight`，`bias`亦然。因为`net[0]`这样根据下标访问子模块的写法只有当`net`是个`ModuleList`或者`Sequential`实例时才可以

```text
#方法一
class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.l1 = nn.Linear(2,1)
        
    def forward(self, x):
        out = self.l1(x)
        return out
    
net = LinearNet()
​
#方法二
net = nn.Sequential(
    nn.Linear(num_inputs, 16)
    # 此处还可以传入其他层
    nn.Linear(16, 1)
    )
```

**两种方法的参数设置**

```text
#Sequential下定义一层: net.xx(层名).xx 
#同时也适用于法一（每层都命名）
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)
​
#Sequential下定义二层: net.xx(layername)[i].xx
init.normal_(net.linearNet[0].weight, mean=0, std=0.01)
init.constant_(net.linearNet[0].bias, val=0)
```

**参数设置原则** 将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。

**学习率设置** 当我们定义了多个不同的子网络时，如果有需要，也可以设置不同的学习率。

```text
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1(如：l1).parameters()}, # lr=0.03
                {'params': net.subnet2(如：l2).parameters(), 'lr': 0.01}
            ], lr=0.03)
print(optimizer)
```

#### softmax与分类模型

\#\#\#\#基础知识 ![](https://img2018.cnblogs.com/blog/1813159/202002/1813159-20200214155456626-103327126.png)

**实现过程**

![](https://img2018.cnblogs.com/blog/1813159/202002/1813159-20200214155531558-1487989356.png)

**学习笔记**

**数据下载** 因为国外网站下载特别慢，所以我直接修改了FashionMNIST的下载地址，修改成了本地，不然总是报错。

```text
train_data = torchvision.datasets.FashionMNIST(
    root='./FashionMNIST',
    download=DOWNLOAD_MNIST,
    train=True,
    transform=transforms.ToTensor()
)
```

**几个概念** 分类准确率：正确预测数量与总预测数量之比。 定义初始化模型：这里迷糊了一下，不知道为什么是\[batch, 784\]，不过\[ x \]好像就是横向的。\[x1, x2, x3,x4 ...\]这样。把784个像素拉长了。这样的话，输入的维度就是宽，就像压扁了一样。

**遇到的问题** 本次模型属于线性模型，中间没有其他的hiddenlayer。

输入为28 \* 28，输出是10，是典型的多分类问题。要学习本次代码中展示样例的方法。

BATCH\_SIZE 取的是256，有60000个数据，回合数是230多。

相较于连续预测不同的是，将数据x\[ 256, 1, 28, 28\]传入net中，输出的是\[256, 10\]的结果，crossentropy的计算是torch内定的。传入的数据维度是\[batch, num\_type\]。

然后将out中的每一行通过softmax转化为和为1的矩阵，再选出每行中值最大的index与真实的y进行匹配，统计每个batch中总共有多少个正确的预测，并记录总数据元素。在一个epoch结束的时候，计算训练数据的准确度。最后的准确率大概是84%左右。

```text
(out.argmax(dim=1) == batch_y).float().sum().item()
sum_train += batch_y.shape[0]
```

#### 多层感知机

**基础知识**

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hidden layer）。隐藏层位于输入层和输出层之间。 \#\#\#\#实现过程 在定义net时，多加几层的Linear，神经元个数可调整。当数据量较小时，防止过拟合问题。

**学习笔记**

**激活函数的选择** ReLu函数是一个通用的激活函数，目前在大多数情况下使用。但是，ReLU函数只能在隐藏层中使用。 用于分类器时，sigmoid函数及其组合通常效果更好。由于梯度消失问题，有时要避免使用sigmoid和tanh函数。 在神经网络层数较多的时候，最好使用ReLu函数，ReLu函数比较简单计算量少，而sigmoid和tanh函数计算量大很多。 在选择激活函数的时候可以先选用ReLu函数如果效果不理想可以尝试其他激活函数。 **感知机小结** 本次的测试代码大部分沿用了多分类问题的代码段。只做了少许的修改。 定义网络层结构：（之前的方法不能说错，但是可能比较适合于CNN吧

```text
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
)
print(net)
```

然后初始化weight和bias

```text
init.normal_(net[1].weight, mean=0, std=0.01)
init.constant_(net[1].bias, val=0)
init.normal_(net[3].weight, mean=0, std=0.01)
init.constant_(net[3].bias, val=0)
```

其他地方未做改动，最后的正确率有86%左右。估计多加几层会好一些。

#### 模型拟合

