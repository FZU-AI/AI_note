# NDArray的基本使用



## 前言

> 本书使用的是MXNet框架，在数据处理方面使用的是NDArray库，此前使用的较多的是Numpy库，所以本次在做一个NDArray和Numpy的知识点集合，比较两者的异同，过于简单的内容就不赘述了



## 数据操作

##### 关于arange()的使用区别

```python
import numpy as np
from mxnet import nd

x=np.arange(12)
y=nd.arange(12)
```

##### 结果可以看到，numpy使用arange()初始化为整数类型，而ndarray初始化为浮点类型

```
[ 0  1  2  3  4  5  6  7  8  9 10 11]

[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]
<NDArray 12 @cpu(0)>
```



##### 关于随机化数组的使用区别

```python
import numpy as np
from mxnet import nd

x1=np.random.normal(0,1,size=(3, 4))
x2=np.random.randint(0,10,size=(3, 4))
y1=nd.random.normal(0,1,shape=(3, 4))
y2=nd.random.randint(0,10,shape=(3, 4))
```

##### 结果可以看到，两者使用几乎一致，唯一的差别点就是size和shape的使用差别了

```
[[ 1.1630785   0.4838046   0.29956347  0.15302546]
 [-1.1688148   1.558071   -0.5459446  -2.3556297 ]
 [ 0.54144025  2.6785064   1.2546344  -0.54877406]]
<NDArray 3x4 @cpu(0)>

[[0 8 3 9]
 [6 3 9 5]
 [3 1 9 9]]
<NDArray 3x4 @cpu(0)>
```



##### 关于nd的简单运算操作，运算的前提是矩阵维度一致

```python
import numpy as np
from mxnet import nd

x1=nd.arange(12).reshape((3,4))
x2=nd.arange(12,24).reshape((3,4))
y1=x1+x2
y2=x1*x2 #注意这是矩阵间对应点相乘
y3=x2/x1
y4=nd.dot(x1,x2.T)  #T表示转置,注意这是矩阵相乘，注意对应的行列数
```

```
[[12. 14. 16. 18.]
 [20. 22. 24. 26.]
 [28. 30. 32. 34.]]
<NDArray 3x4 @cpu(0)>

[[  0.  13.  28.  45.]
 [ 64.  85. 108. 133.]
 [160. 189. 220. 253.]]
<NDArray 3x4 @cpu(0)>

[[       inf 13.         7.         5.       ]
 [ 4.         3.4        3.         2.7142856]
 [ 2.5        2.3333333  2.2        2.090909 ]]
<NDArray 3x4 @cpu(0)>

[[ 86. 110. 134.]
 [302. 390. 478.]
 [518. 670. 822.]]
<NDArray 3x3 @cpu(0)>

```



##### 广播机制，两个库均含有，表示当维度不一致时，低纬度的整合与高纬度中

```python
import numpy as np
from mxnet import nd

x1=nd.arange(12).reshape((3,4)) #三行四列
x2=nd.arange(20,24) #一行三列
y=x1+x2
```

```
[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 3x4 @cpu(0)>

[20. 21. 22. 23.]
<NDArray 4 @cpu(0)>

[[20. 22. 24. 26.]  
 [24. 26. 28. 30.]
 [28. 30. 32. 34.]]
<NDArray 3x4 @cpu(0)>
```



##### NDArray和NumPy相互变换

```python
import numpy as np
from mxnet import nd

x=np.arange(12).reshape((3,4))
y=nd.array(x) #转化为nd类型
z=y.asnumpy() #又转化为np类型
```

```
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
<NDArray 3x4 @cpu(0)>

[[ 0.  1.  2.  3.]
 [ 4.  5.  6.  7.]
 [ 8.  9. 10. 11.]]
```

