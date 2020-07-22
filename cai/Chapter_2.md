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

a=nd.zeros((2,3,4))		//创建张量a，可以将a看做由2页纸，每一页纸有一个2行4列的数组构成

```
[[[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]

 [[0. 0. 0. 0.]
  [0. 0. 0. 0.]
  [0. 0. 0. 0.]]]
```

##### 6.通过Python的list（列表）来创建特定的NDArray中每个元素的值

d=nd.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
d

```
[[ 1.  2.  3.  4.]
 [ 5.  6.  7.  8.]
 [ 9. 10. 11. 12.]]
```

##### 7.随机生成NDArray中每个元素的值

如：创建一个形状为（4,3）的NDArray，且每个元素都服从均值为0，标准差为1的正态分布

c=nd.random.normal(0,1,shape=(4,3))
c

```
[[-0.5916499   0.85860497 -0.22794183]
 [ 0.20131476  0.3500547   0.5360521 ]
 [ 1.5194443   1.9040879  -1.5734432 ]
 [-0.14007866  0.29670075  1.3111951 ]]
```

## 2.3 NDArray的运算

##### 1.加法

如：① x=nd.arange(12)
            X=x.reshape((3,4))
	        X

```
[[11. 12. 13.  4.]
 [15.  6.  7. 18.]
 [19. 10. 11. 12.]]
 
```

##### 2.乘法

##### 3.除法

##### 4.指数运算