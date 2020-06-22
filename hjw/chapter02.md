# 1.mxnet张量的基本知识

```python
from mxnet import nd    #引入mxnet库
x = nd.arange(12)       #创建一个从0开始的12个连续整数  
x
```

输出结果：

> ```base
> [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
> <NDArray 12 @cpu(0)>
> ```

------

```python
x.shape                #获取形状，即几行几列
x.size                #获取元素个数
x.reshape((3,4))      #修改形状为3行4列
nd.zeros((2,3,4))    #生成元素全为0的指定形状的张量
nd.random.normal(0,1,shape=(2,3))         #生成元素都随机采样于均值为0，标准差为1的正态分布，形状为（2,3）
y = nd.array([[2,1,4,3],[1,2,3,4],[4,3,2,1]])   #利用python中的列表创建张量
nd.ones((3,4))     #生成元素全为1的指定形状的张量
```

其基本操作和numpy中的基本一致

