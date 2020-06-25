# **第一章**

## 1.机器学习与深度学习

通俗来说，机器学习讨论各式各样的适用于不同问题的函数形式，以及如何使用数据来有效的获取函数参数值。而深度学习是机器学习中的一类函数，他们的形式通常为多层神经网络。

## 2.神经网络

- 交替使用线性处理单元和非线性单元，它们通常被称为层
- 使用链式法则（即反向传播）来更新网络参数

# 第二章

## 1.环境配置

win10+python3+mxnet101

用conda进行包管理

CUDA各版本下载地址：https://developer.nvidia.com/cuda-toolkit-archive

英伟达显卡驱动下载：https://www.nvidia.cn/Download/index.aspx?lang=cn#

mxnet直接用condaPrompt：

```
pip install mxnet-cu101
```



> PS:本来要用vmware+ubuntu+python3，结果弄了一天都没弄好，各种各样的错误，于是放弃



## 2.数据操作

1. 对两个形状不同的array做元素运算会触发广播机制（适当复制元素使之形状相同）

   不过自己用3行2列与4行1列的array相加直接报错了，似乎要特定形状才会触发

2. 索引：比如x[1：3]是行索引，下标从0开始，遵循左闭右开的惯例，相当于索引第2行第3行

3. 内存开销：一个新变量被进行赋值操作会开辟新的内存地址，旧变量则不会

## 3.查阅文档

- 查找模块里所有函数

```
from mxnet import nd

print(dir(nd.random))
```

- 查找特定函数

```
help(nd.arange)
```

