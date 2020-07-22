# 第五章

### 5.1  二维卷积层

虽然卷积层得名于卷积运算，但是我们通常在卷积层中使用更加直观的互相关运算。在二维互相关运算中，卷积窗口从输入数组的最左上方开始，按从左往右、从上向下的顺序，依次在输入组上滑动

![image-20200712192431594](https://github.com/FZU-AI/AI_note/blob/master/woo/deep_learning/photo/5.1.png)
$$
0 * 0+1 * 1+3 * 2+4 * 3=19\\

1 * 0+2 * 1+4 * 2+5 * 3=25\\

3 * 0+4 * 1+6 * 2+7 * 3=37\\

4 * 0+5 * 1+7 * 2+8 * 3=43\\
$$

### 5.2  填充和步幅

#### 5.2.1  填充

填充是指在输入的高、宽两侧填充元素（一般是0），经常适当填充使输入和输出的形状相同。在卷积神经网络中经常用奇数高宽的卷积核，那么在填充的时候两端都是相等的个数。



#### 5.2.2  步幅

卷积窗口在卷积运算中，从输入数组的最上方开始在输入数组上滑动，我们称每次滑动的行数和列数为步幅。



### 5.3  多输入/出通道

之前两节都是h*w的二维数组，但真实数据的维度经常更高，比如彩色图像：除了高和宽，还有RGB三个颜色通道，可以用3 * h*w的多维数组表示，我们将3这一维称为通道。



### 5.4  池化层

之前的边缘检测输入和卷积核都是已知的，但实际情况中图像位置不固定，对模式识别造成不便。池化层的提出是为了缓解卷积层对位置的过度敏感性。

池化层不同于之前的互相关计算，而是直接计算池化窗口内的最大值/最小值/平均值



### 5.5  卷积神经网络（LeNet）

LeNet分为卷积层块和全连接层两部分。

卷积层块的基本单位是卷积层后接最大池化层：卷积层用来识别图像里的空间模式，如线条和物体局部，最大池化层用来降低卷积层对位置的敏感性。

全连接层块含3个全连接层，输出个数分别为120、84、10。

#### 5.5.1  建立模型

```
import d2lzh as d2l
import mxnet as mx
from mxnet import autograd,gluon,init,nd
from mxnet.gluon import loss as gloss,nn
import time 

net=nn.Sequential()
net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Conv2D(channels=16,kernel_size=5,activation='sigmoid'),
        nn.MaxPool2D(pool_size=2,strides=2),
        nn.Dense(120,activation='sigmoid'),
        nn.Dense(84,activation='sigmoid'),
        nn.Dense(10))
        
X=nd.random.uniform(shape=(1,1,28,28))
net.initialize()
for layer in net:
    X=layer(X)
    print(layer.name,'output shape:\t',X.shape)

```

#### 5.5.2   获取数据和训练模型

```
batch_size =256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)

#定义函数以GPU计算
def try_gpu():
    try:
        ctx=mx.gpu()
        _ =nd.zeros((1,),ctx=ctx)
    except mx.base.MXNetError:
        ctx=mx.cpu()
    return ctx

ctx=try_gpu()

#定义一个评估准确率的函数
def evaluate_accuracy(data_iter,net,ctx):
    acc_sum,n=nd.array([0],ctx=ctx),0
    for X,y in data_iter:
        X,y = X.as_in_context(ctx),y.as_in_context(ctx).astype('float32')
        acc_sum+=(net(X).argmax(axis=1)==y).sum()
        n+=y.size
    return acc_sum.asscalar()/n

#训练函数
def train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs):
    print('在',ctx,'上训练')
    #使用交叉熵损失函数
    loss=gloss.SoftmaxCrossEntropyLoss()
    #训练次数为num_epochs
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n,start=0.0,0.0,0,time.time()
        for X,y in train_iter:
            X,y = X.as_in_context(ctx),y.as_in_context(ctx)
            with autograd.record():
                y_hat=net(X)
                l=loss(y_hat,y).sum()
            #计算梯度
            l.backward()
            #更新模型参数
            trainer.step(batch_size)
            y=y.astype('float32')
            train_l_sum+=l.asscalar()
            train_acc_sum+=(y_hat.argmax(axis=1)==y).sum().asscalar()
            n+=y.size
        test_acc=evaluate_accuracy(test_iter,net,ctx)
        print('epoch%d,loss%.4f,train acc%.3f,test acc%.3f,time%.1f s'
              %(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc,time.time()-start)
        )

#测试
lr,num_epochs=0.9,5
#初始化神经网络
net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
#设置训练时方法
trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
```

### 5.6  深度卷积神经网络AlexNet

#### 5.6.1  AlexNet

AlexNet有八层神经网络，包含5层卷积和2层全连接隐藏层，以及一个全连接输出层。

第一层的卷积窗口形状是11 X  11，第二层为5  x  5，之后全采用3  x  3，此外，第1、3、5个卷积层都使用了3  x  3步幅为2的最大池化层，而且卷积通道数也是LeNet的数十倍。紧接着最后一个卷积层的是两个输出个数为4096的全连接层，这两个巨大的全连接层带来将近1GB的模型参数。不仅如此，AlexNet还对激活函数、初始化方法等等技术做了改进。



#### 5.6.2  构造神经网络

```
import d2lzh as d2l
from mxnet import gluon,init,nd
from mxnet.gluon import data as gdata,nn
import os
import sys

net=nn.Sequential()
net.add(nn.Conv2D(96,kernel_size=11,strides=4,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Conv2D(256,kernel_size=5,padding=2,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
        nn.Conv2D(256,kernel_size=3,padding=1,activation='relu'),
        nn.MaxPool2D(pool_size=3,strides=2),
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
        nn.Dense(10))
```

#### 5.6.3  读取数据

使用前面的Fashion-MNIST数据集来演示AlexNet，读取数据的时候额外将图像高和宽扩大到224

```
def load_data_fashion_mnist(batch_size,resize=None,root=os.path.join('~','.mxnet','datasets','fashion_mnist')):
    root=os.path.expanduser(root)
    transformer=[]
    if resize:
        transformer+=[gdata.vision.transforms.Resize(resize)]
    transformer+=[gdata.vision.transforms.ToTensor()]
    transformer=gdata.vision.transforms.Compose(transformer)
    mnist_train=gdata.vision.FashionMNIST(root=root,train=True)
    mnist_test=gdata.vision.FashionMNIST(root=root,train=False)
    num_workers=0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transormer),batch_size,shuffle=True,num_workers=num_workers
    )
    test_iter = gdata.DataLoader( 
        mnist_test.transform_first(transformer), batch_size, shuffle=False, num_workers=num_workers
    ) 
    return train_iter, test_iter

batch_size = 128 
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224) 
```

#### 5.6.4  训练数据

```
lr, num_epochs, ctx = 0.01, 5, d2l.try_gpu() 
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier()) 
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr}) 
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs) 
```

### 5.7  使用重复元素的网络（VGG）

#### 5.7.1  VGG块

VGG块的组成规律是：连续使用个数相同的填充为1，窗口形状为3  x  3的卷积层后接上一个步幅为2，窗口形状为2  x  2的最大池化层。卷积层保持输入的高和宽不变，而池化层对其减半。

```
import d2lzh as d2l
from mxnet import gluon,init,nd
from mxnet.gluon import nn

#指定卷积层数量和输出通道数
def vgg_block(num_convs,num_channels):
	#初始化容器
    blk=nn.Sequential()
    #加入指定个数的卷积层
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels,kernel_size=3,
                          padding=1,activation='relu'))
    #最后加入一个池化层                    
    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
    return blk
```

#### 5.7.2  VGG网络

由8个卷积层和3个全连接层构成

```
conv_arch = ((1,64),(1,128),(2,256),(2,512),(2,512))
#生成VGG网络
def vgg(conv_arch):
    net=nn.Sequential()
    #卷积层部分，每次生成一个VGG块
    for (num_convs,num_channels) in conv_arch:
        net.add(vgg_block(num_convs,num_channels))        
    #全连接层部分
    net.add(
            nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
            nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
            nn.Dense(10)
    )
    return net

net = vgg(conv_arch)
```

#### 5.7.3  获取数据和训练模型

```
ratio=4
#构造一个较小的网络
small_conv_arch=[(pair[0],pair[1]//ratio) for pair in conv_arch]
net=vgg(small_conv_arch)

lr,num_epochs,batch_size,ctx=0.05,5,128,d2l.try_gpu()
net.initialize(ctx=ctx,init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224) 
d2l.train_ch5(net,train_iter, test_iter,batch_size,trainer,ctx,num_epochs)
```

### 5.8  网络中的网络（NiN）

NiN不同于之前的LeNet、AlexNet、VGG，而NiN由多个小网络构成一个深层网络。

![image-20200717223752724](https://github.com/FZU-AI/AI_note/blob/master/woo/deep_learning/photo/5.7.png)



### 5.9  批量归一化

通常来说，数据标准化预处理对于浅层模型足够有效了，但是对于深层神经网络容易在靠近输入层的时候剧烈变化。批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使各层数值更加稳定。



### 5.10  残差网络ResNet

在实践中，添加过多的层会导致训练误差不降反升，即使用了批量归一化也依然存在该问题。残差网络就是为了解决这个问题。



### 5.11  稠密连接网络DenseNet

DenseNet的主要构建模块是稠密块和过渡层，前者定义了输入和输出是如何连接的，后者则用来控制通道数使之不过大。

























