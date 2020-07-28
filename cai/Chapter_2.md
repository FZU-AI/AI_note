

# 第二章 配置环境和使用NDArray

## 2.1 环境搭建

#### Windows用户

##### 1.根据操作系统安装Miniconda

参考(https://blog.csdn.net/GhostGuest/article/details/104471272?biz_id=102&utm_term=miniconda%E5%AE%89%E8%A3%85%E6%95%99%E7%A8%8B&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-0-104471272&spm=1018.2118.3001.4187)

##### 2.下载本书所需的代码压缩包

在浏览器中输入https://zh.d2l.ai/d2l-zh.zip

##### 3.使用conda创建虚拟环境；

参考（https://blog.csdn.net/LLM1602/article/details/105280652?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase）

##### 4.激活之前创建的环境，输入conda activate gluon

(注：可能显示无法激活，一个原因是因为找不到激活的gluon地址

解决方法：在电脑中找到Miniconda的安装位置，进一步寻找envs/gluon，最后输入conda activate 此位置即可)

##### 5.打开Jupyter记事本



## 2.2 使用NDArray

##### 1.从MXNet导入NDArray模块

`from mxnet import nd     //导入nd`

##### 2.用nd创建一个数组

① `a=nd.zeros(12)		//a是一个有12个元素的一维数组，其值均为0`

​     `b=nd.ones(12)		//b是一个有12个元素的一维数组，其值均为1`

② `a=nd.zeros(（3,4）)		//a是一个3行4列的二维数组`

​												`//注意：两者的区别，创建二维及以上的数组，有两重括号`

​												`//不能是a=nd.zeros(3,4)`

##### 3.使用size测量出数组的元素总个数  /  使用shape来获取数组的形状  /  使用reshape重塑数组的形状

① `c.size`		//若c是一个2行5列的二维数组，则①的值为10，②的值为（2,5）`

②`c.shape`

③`c.reshape((5,2))`		//③的形状变成5行2列

##### 4.使用arange（）

arange(start,end,step)：创建一个范围从start到end（不包括end）的一维数组，步长为step，step默认1。

##### 5.张量跟数组/向量的关系，创建张量

向量和数组是一种特殊的张量

```
a=nd.zeros((2,3,4))		//创建张量a，可以将a看做由2页纸，每一页纸有一个2行4列的数组构成
```

```
[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]]
```

##### 6.通过Python的list（列表）来创建特定的NDArray中每个元素的值

```
d=nd.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
d
```

```
[[ 1.  2.  3.  4.]
 [ 5.  6.  7.  8.]
 [ 9. 10. 11. 12.]]
```

##### 7.随机生成NDArray中每个元素的值

如：创建一个形状为（4,3）的NDArray，且每个元素都服从均值为0，标准差为1的正态分布

```
c=nd.random.normal(0,1,shape=(4,3))
c
```

```
[[-0.5916499   0.85860497 -0.22794183]
 [ 0.20131476  0.3500547   0.5360521 ]
 [ 1.5194443   1.9040879  -1.5734432 ]
 [-0.14007866  0.29670075  1.3111951 ]]
```

## 2.3 NDArray的运算

##### 1.加法

如：

```
 x=nd.arange(12)
 X=x.reshape((3,4))
 X
```

```
 [[11. 12. 13.  4.]
 [15.  6.  7. 18.]
 [19. 10. 11. 12.]]
```

```
Y=nd.array([[11,12,13,4],[15,6,7,18],[19,10,11,12]])
Y
```

```
[[11. 12. 13.  4.]
 [15.  6.  7. 18.]
 [19. 10. 11. 12.]]
```

执行X+Y

```
[[11. 13. 15.  7.]
 [19. 11. 13. 25.]
 [27. 19. 21. 23.]]
```

##### 2.乘法

执行 X*Y

```
[[  0.  12.  26.  12.]
 [ 60.  30.  42. 126.]
 [152.  90. 110. 132.]]
```

##### 3.除法

执行 X/Y

```
[[0.         0.08333334 0.15384616 0.75      ]
 [0.26666668 0.8333333  0.85714287 0.3888889 ]
 [0.42105263 0.9        0.90909094 0.9166667 ]]
```

##### 4.指数运算

执行Y.exp()

```
[[5.9874141e+04 1.6275480e+05 4.4241341e+05 5.4598148e+01]
 [3.2690172e+06 4.0342880e+02 1.0966332e+03 6.5659968e+07]
 [1.7848230e+08 2.2026465e+04 5.9874141e+04 1.6275480e+05]]
```

##### 5.X与Y的转置(Y.T)做矩阵乘法运算 : X为n行m列，Y为m行n列。注意原先矩阵X*Y是不能相乘的

```
[[ 50.  74.  68.]
 [210. 258. 276.]
 [370. 442. 484.]]
```

##### 6.使用concat将多个NDArray连结（concatenate）

nd.concat(X, Y, dim=0)：按行连结两个矩阵

  nd.concat(X, Y, dim=1)：按列连结两个矩阵

执行 nd.concat(X, Y, dim=0),nd.concat(X, Y, dim=1)

```
 [[ 0.  1.  2.  3.]
  [ 4.  5.  6.  7.]
  [ 8.  9. 10. 11.]
  [11. 12. 13.  4.]
  [15.  6.  7. 18.]
  [19. 10. 11. 12.]]
 <NDArray 6x4 @cpu(0)>,
 
 [[ 0.  1.  2.  3. 11. 12. 13.  4.]
  [ 4.  5.  6.  7. 15.  6.  7. 18.]
  [ 8.  9. 10. 11. 19. 10. 11. 12.]]
   <NDArray 3x8 @cpu(0)>)
```

## 2.4 广播机制

###### 当对两个形状不同的NDArray按元素运算时，可能会触发⼴播（broadcasting）机制：先适当复制元素使这两个NDArray形状相同后再按元素运算。

```
A = nd.arange(3).reshape((3, 1))
B = nd.arange(2).reshape((1, 2))
A,B
```

```
(
 [[0.]
  [1.]
  [2.]]
 <NDArray 3x1 @cpu(0)>,
 
 [[0. 1.]]
 <NDArray 1x2 @cpu(0)>)
```

###### 执行A+B     //由于A和B分别是3⾏1列和1⾏2列的矩阵，如果要计算A + B，那么A中第⼀列的3个元素被⼴播（复制）到了第⼆列，而B中第⼀⾏的2个元素被⼴播（复制）到了第⼆⾏和第三⾏。如此，就可以对2个3⾏2列的矩阵按元素相加。

```
[[0. 1.]
 [1. 2.]
 [2. 3.]]
<NDArray 3x2 @cpu(0)>
```

## 2.5 索引

###### 索引在NDArray中的作用:寻找特定元素的位置

###### NDArray的索引值从0开始，即一个n行m列的二维数组其索引可为：0-n-1（行索引）

X矩阵如下：

 [[11. 12. 13.  4.]
 [15.  6.  7. 18.]
 [19. 10. 11. 12.]]

① 执行 X[1:3]		//截取出矩阵X的第2,3行的元素

```
[[ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 2x4 @cpu(0)>
```

② 执行 X[1，3]=20		//将矩阵第2行第4列的元素7变为20，注是逗号，不是冒号

```
[[ 0.  1.  2.  3.]
 [ 4.  5.  6. 20.]
 [ 8.  9. 10. 11.]]
```

## 2.6 运算的内存开销

###### 每次进行数据操作都会占用内存空间，来存放运算的结果。

比如原先已有矩阵X、Y，之后执行Y=X+Y的运算，这个最终的Y值存放的内存空间与之前的Y的内存空间不一致（可以使用id函数进行验证）

```
before = id(Y)

Y = Y + X

id(Y) == before
```

输出

```
False
```

那么如何让最终Y的内存空间保持一致，节省空间？

（1）索引方法：过zeros_like创建和Y形状相同且元素为0的NDArray，记为Z。接下来，我们把X +

Y的结果通过[:]写进Z对应的内存中。

```
Z = Y.zeros_like()
before = id(Z)
Z[:] = X + Y
id(Z) == before
```

```
True
```

实际上，例中我们还是为X+Y开了临时内存来存储计算结果，再复制到Z对应的内存。如果想避免这个临时内存开销，我们可以使用运算符全名函数中的out参数。

```
nd.elemwise_add(X, Y, out=Z)
id(Z) == before
```

```
True
```

## 2.7 NumPy和NDArray之间的互换

①使用array（）将NumPy变成NDArray

```
import numpy as np
P = np.ones((2, 3))
D = nd.array(P)
D
```

```
[[1. 1. 1.]
 [1. 1. 1.]]
<NDArray 2x3 @cpu(0)>
```

②使用asnumpy（）将NDArray变成NumPy

```
D.asnumpy()
```

```
array([[1., 1., 1.],
       [1., 1., 1.]], dtype=float32)
```

## 2.8 自动求梯度

### 使用MXnet自动求梯度的步骤：

思想：利用autograd来自动求解梯度

1.从mxnet中导入autograd模块；

```
from mxnet import autograd

from mxnet import nd
```

2.生成n维的列向量，如n=4

```
x = nd.arange(4).reshape((4,1))
```

```
[[0.]
 [1.]
 [2.]
 [3.]]
<NDArray 4x1 @cpu(0)>
```

3.使用attach_grad()函数来申请存储梯度所需的内存，同时因为默认条件下mxnet不会记录梯度求解过程，因此需要调用record()函数来记录相关过程

```
x.attach_grad()
with autograd.record():
    y = 2 * nd.dot(x.T, x)
```

4. 调用backward()函数来自动求梯度,得到y关于x的梯度

```
y.backward()
x.grad
```

```
[[ 0.]
 [ 4.]
 [ 8.]
 [12.]]
<NDArray 4x1 @cpu(0)>
```

#### 2.8.1 训练模式和预测模式

在调⽤record函数后，MXNet会记录并计算梯度。此外，默认情况下autograd还会将运⾏模式从预测模式转为训练模式；这可以通过调⽤is_training函数来查看。

```
print(autograd.is_training())

with autograd.record():

print(autograd.is_training())
```



#### 2.8.2 对Python控制流求梯度

## 2.9 查阅文档

#### （1）使用dir（）函数查看一个模块提供的函数和类

例如，查看NDArray的函数功能

```
from mxnet import nd

print(dir(nd.random))
```

```
['NDArray', '_Null', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_internal', '_random_helper', 'current_context', 'exponential', 'exponential_like', 'gamma', 'gamma_like', 'generalized_negative_binomial', 'generalized_negative_binomial_like', 'multinomial', 'negative_binomial', 'negative_binomial_like', 'normal', 'normal_like', 'numeric_types', 'poisson', 'poisson_like', 'randint', 'randn', 'shuffle', 'uniform', 'uniform_like']
```

#### （2）使用help（）函数，具体了解某个函数或类的用法

以NDArray中的ones_like（）函数为例，查阅它的⽤法

```
help(nd.ones_like)
```

```
Help on function ones_like:

ones_like(data=None, out=None, name=None, **kwargs)
    Return an array of ones with the same shape and type
    as the input array.
    
    Examples::
    
      x = [[ 0.,  0.,  0.],
           [ 0.,  0.,  0.]]
    
      ones_like(x) = [[ 1.,  1.,  1.],
                      [ 1.,  1.,  1.]]
    
    
    
    Parameters
    ----------
    data : NDArray
        The input
    
    out : NDArray, optional
        The output NDArray to hold the result.
    
    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
```

61+11+11