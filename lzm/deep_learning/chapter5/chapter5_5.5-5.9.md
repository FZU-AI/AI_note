**5.5 卷积神经网络（LeNet）**

MLP对高和宽均是n像素的图像进行分类，是将图像中的像素逐行展开，得到长度为n<sup>2</sup>的向量，并输入进全连接层中。然而这种分类方法有一定的局限性。  

1.图像在同一列邻近的像素在这个向量中可能相距较远。它们构成的模式可能难以被模型识别。
 
2.对于大尺寸的输入图像，使用全连接层容易导致模型过大。假设输入是高和宽均为1,000像素的彩色照片（含3个通道）。即使全连接层输出个数仍是256，该层权重参数的形状也是3,000,000 × 256：它占用了大约3 GB的内存或显存。这会带来过于复杂的模型和过高的存储开销。  

卷积神经网络（CNN）就是含卷积层的网络。卷积层解决了以上这两个问题。  
一方面，卷积层保留输入形状，使图像的像素在高和宽两个方向上的相关性均可能被有效识别；  
另一方面，卷积层通过滑动卷积窗口将同⼀卷积核与不同位置的输入重复计算，从而避免参数尺寸过大。 

**1. LeNet模型** 

LeNet交替使用卷积层和最大池化层后接全连接层来进行图像分类。  
LeNet分为卷积层块和全连接层块两个部分。  
1.卷积层块里的基本单位是卷积层后接最大池化层：卷积层用来识别图像里的空间模式，如线条和物体局部；之后的最大池化层则用来降低卷积层对位置的敏感性。卷积层块由两个这样的基本单位重复堆叠构成。  
在卷积层块中，每个卷积层都使用5×5的窗口，并在输出上使用sigmoid激活函数。第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。这是因为第二个卷积层比第一个卷积层的输入的高和宽要小，所以增加输出通道使两个卷积层的参数尺寸类似。卷积层块的两个最大池化层的窗口形状均为2×2，且步幅为2。由于池化窗口与步幅形状相同，池化窗口在输入上每次滑动所覆盖的区域互不重叠。   
2.卷积层块的输出形状为(批量大小,通道,高,宽)。当卷积层块的输出传入全连接层块时，全连接层块会将小批量中每个样本变平（flatten）。也就是说，全连接层的输入形状将变成二维，其中第一维是小批量中的样本数，第二维是每个样本变平后的向量表示，且向量长度为通道、高和宽的乘积。全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。 

	# 通过Sequential类来实现LeNet模型。
	import d2lzh as d2l
	import mxnet as mx
	from mxnet import autograd,gluon,init,nd
	from mxnet.gluon import loss as gloss,nn
	import time
	
	net=nn.Sequential()
	net.add(nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),# 卷积层
	       nn.MaxPool2D(pool_size=2,strides=2),# 最大池化层
	       nn.Conv2D(channels=6,kernel_size=5,activation='sigmoid'),
	       nn.MaxPool2D(pool_size=2,strides=2),
	       # Dense会默认将(批量大小, 通道, 高, 宽)形状的输⼊转换成(批量大小, 通道*⾼*宽)形状的输入
	       nn.Dense(120,activation='sigmoid'),# 全连接层
	       nn.Dense(84,activation='sigmoid'),
	       nn.Dense(10)# 输出层
	       )
	
卷积层由于使用高和宽均为5的卷积核，从而将高和宽分别减小4（-5+1），而池化层则将高和宽减半（步幅为2），但通道数则从1增加到16。全连接层则逐层减少输出个数，直到变成图像的类别数10。 
	
**2. 获取数据和训练模型**

获取训练集和测试集

	batch_size=256
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)

使用GPU来加速计算。

	# 尝试在gpu(0)上 创建NDArray，如果成功则使用gpu(0)，否则仍然使用CPU。 
	# 本函数已保存在d2lzh包中方便以后使用 
	def try_gpu():
	    try:
	        ctx=mx.gpu()
	        _=nd.zeros((1,),ctx=ctx)# 尝试在gpu上创建NDArray
	    except mx.base.MXNetError:
	            ctx =mx.cpu()
	    return ctx
	
	# 本函数已保存在d2lzh包中方便以后使用。
	def evaluate_accuracy(data_iter,net,ctx):
	    acc_sum,n=nd.array([0],ctx=ctx),0
	    for X,y in data_iter:
	        # 如果目标变量和源变量的设备不同，则复制数据到ctx上 # 如果目标变量和源变量的设备相同，则目标变量共享源变量的内存或显存
	        X,y=X.as_in_context(ctx),y.as_in_context(ctx).astype('float32')
	        acc_sum+=( net(X).argmax(axis=1)==y ).sum()# argmax(axis=1)返回每行最大元素的索引
	        n+=y.size
	    return acc_sum.asscalar()/n
	
	# 本函数已保存在d2lzh包中方便以后使用
	def train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs):
	    print('training on',ctx)
	    loss=gloss.SoftmaxCrossEntropyLoss()
	    for epoch in range (num_epochs):
	        train_l_sum,train_acc_sum,n,start=0.0,0.0,0,time.time()
	        for X,y in train_iter:
	            X,y=X.as_in_context(ctx),y.as_in_context(ctx)# 将数据复制到相应的设备上
	            with autograd.record():
	                y_hat=net(X)
	                l=loss(y_hat,y).sum() #小批量样品的损失函数和
	            l.backward()
	            trainer.step(batch_size)
	            y=y.astype('float32')
	            train_l_sum+=l.asscalar()# 所有样品的损失函数的和
	            train_acc_sum+=( y_hat.argmax(axis=1)==y ).sum().asscalar()# 所有样品预测准确的个数
	            n+=y.size# 样本数
	        test_acc=evaluate_accuracy(test_iter,net,ctx)
	        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f,time %.lf sec'
	              %(epoch+1,train_l_sum/n,train_acc_sum/n,test_acc,time.time()-start))

重新将模型参数初始化到设备变量ctx之上，并使用Xavier随机初始化。损失函数和训练算法则依然使用交叉熵损失函数和小批量随机梯度下降。

	ctx=mx.cpu()
	lr,num_epochs=0.9,5
	net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)                      

**5.6 深度卷积神经网络（AlexNet）** 

**1. 学习特征表示** 

多层神经网络可以学得数据的多级表征，并逐级表示越来越抽象的概念或模式。输入的逐级表示由多层模型中的参数决定，而这些参数都是学出来的。

**2. AlexNet模型**

AlexNet跟LeNet结构类似，但使⽤了更多的卷积层和更⼤的参数空间来拟合⼤规模数据集ImageNet。它是浅层神经网络和深度神经网络的分界线。

AlexNet与LeNet的区别：  
1.与相对较小的LeNet相比，AlexNet包含8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。  
AlexNet第一层中的卷积窗口形状是11 × 11。第二层中的卷积窗口形状减小到5×5，之后全采用3×3。此外，第一、第二和第五个卷积层之后都使用了窗口形状为3×3、步幅为2的最⼤池化层。而且，AlexNet使用的卷积通道数也⼤于LeNet中的卷积通道数数⼗倍。紧接着最后⼀个卷积层的是两个输出个数为4096的全连接层。  
2.AlexNet将sigmoid激活函数改成了更加简单的ReLU激活函数。ReLU激活函数的计算更简单，在不同的参数初始化方法下使模型更容易训练。若模型参数初始化不当，sigmoid函数可能在正区间得到几乎为0的梯度，从而令模型无法得到有效训练。   
3.AlexNet通过丢弃法来控制全连接层的模型复杂度。而LeNet并没有使用丢弃法。   
4.AlexNet引入了大量的图像增广，如翻转、裁剪和颜⾊变化，从而进一步扩大数据集来缓解过拟合。

	# 实现AlexNet模型
	import d2lzh as d2l
	from mxnet import gluon,init,nd
	from mxnet.gluon import data as gdata,nn
	import os
	import sys
	
	net=nn.Sequential()
	# 使用较大的11 x 11窗口来捕获物体。同时使⽤步幅4来较⼤幅度减小输出高和宽。这⾥使⽤的输出通 
	# 道数比LeNet中的也要大很多 
	net.add(nn.Conv2D(96,kernel_size=11,strides=4,activation='relu'),#(n-k+p+s)/s
	       nn.MaxPool2D(pool_size=3,strides=2),
	       # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数 
	       nn.Conv2D(256,kernel_size=5,padding=2,activation='relu'),
	       nn.MaxPool2D(pool_size=3,strides=2),
	       # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
	       # 前两个卷积层后不使用池化层来减小输入的高和宽 
	       nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
	       nn.Conv2D(384,kernel_size=3,padding=1,activation='relu'),
	       nn.Conv2D(256,kernel_size=3,padding=1,activation='relu'),
	       nn.MaxPool2D(pool_size=3,strides=2), 
	       # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合 
	       nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
	       nn.Dense(4096,activation='relu'),nn.Dropout(0.5), 
	       # 输出层。由于这里使用Fashion-MNIST，所以类别数为10，而非论⽂中的1000 
	       nn.Dense(10)
	       )
	
**3. 读取数据**

	# 本函数已保存在d2lzh包中方便以后使用
	# os.path.join('~','.mxnet','datasets','fashion-mnist')在逗号之间添加路径分隔符\\
	def load_data_fashion_mnist(batch_size,resize=None,root=os.path.join('~','.mxnet','datasets','fashion-mnist')):
	    root=os.path.expanduser(root) # 将'~'展开为用户路径
	    transformer=[]
	    if resize:# Resize实例将图像设置为给定大小。
	        transformer+=[gdata.vision.transforms.Resize(resize)]
	    transformer+=[gdata.vision.transforms.ToTensor()]
	    # 用Compose实例来将这两个(Resize和Totensor)变换串联
	    transformer=gdata.vision.transforms.Compose(transformer)
		# 第⼀次调⽤时会自动从⽹上获取数据保存在root路径，若不指定root，则保存在默认路径。
	    mnist_train=gdata.vision.FashionMNIST(root=root,train=True)# 训练集
	    mnist_test=gdata.vision.FashionMNIST(root=root,train=False)# 测试集
	    num_workers=0 if sys.platform.startswith('win32') else 4
	    # transform_first将transformer的变换应用在数据样本的第一个的元素即图像上
	    # gdata.DataLoader读取batch_size大小的数据样本
	    train_iter=gdata.DataLoader(mnist_train.transform_first(transformer),batch_size,shuffle=True,num_workers=num_workers)
	    test_iter=gdata.DataLoader(mnist_test.transform_first(transformer),batch_size,shuffle=False,num_workers=num_workers)
	    return train_iter,test_iter
	
	batch_size=128
	train_iter,test_iter=load_data_fashion_mnist(batch_size,resize=224)
	
**4. 训练**

与LeNet相比，使用了更小的学习率。 

	lr,num_epochs,ctx=0.01,5,d2l.try_gpu()
	net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

**5.7 使用重复元素的网络（VGG）**

VGG提出了可以通过重复使用途简单的基础块来构建深度模型的思路。 

**1. VGG块**

VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为3×3的卷积层后接上⼀个步幅为2、窗口形状为2×2的最大池化层。卷积层保持输入的高和宽不变，而池化层则对其减半。
	
	import d2lzh as d2l
	from mxnet import gluon,init,nd
	from mxnet.gluon import nn

	# vgg_block函数实现VGG块，它可以指定卷积层的数量num_convs和输出通道数num_channels。
	def vgg_block(num_convs,num_channels):
	    blk=nn.Sequential()
	    for _ in range(num_convs):
	        # 卷积层保持输入的高和宽不变(n-3+2+1)
	        blk.add(nn.Conv2D(num_channels,kernel_size=3,padding=1,activation='relu'))
	    # 池化层则对高和宽减半(n-2+2)/2
	    blk.add(nn.MaxPool2D(pool_size=2,strides=2))
	    return blk
	
**2. VGG网络** 

与AlexNet和LeNet一样，VGG网络由卷积层模块后接全连接层模块构成。卷积层模块串联数个`vgg_block`（VGG块），其超参数由变量`conv_arch`定义。该变量指定了每个VGG块里卷积层个数和输出通道数。全连接模块则与AlexNet中的一样。

VGG-11包含了8个卷积层和3个全连接层。它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。第一块的输出通道是64，之后每次对输出通道数翻倍，直到变为512。

	conv_arch=((1,64),(1,128),(2,256),(2,512),(2,512))
	# VGG-11
	def vgg(conv_arch):
	    net=nn.Sequential()
	    # 卷积层部分
	    for (num_convs,num_channels) in conv_arch:
	        net.add(vgg_block(num_convs,num_channels))
	    # 全连接层部分
	    net.add(nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
	           nn.Dense(4096,activation='relu'),nn.Dropout(0.5),
	           nn.Dense(10))
	    return net
	
**3. 获取数据和训练模型** 

与AlexNet相比使用了稍大些的学习率，模型训练过程与AlexNet类似。

	net=vgg(conv_arch)
	import mxnet as mx
	lr,num_epochs,batch_size,ctx=0.05,5,128,mx.cpu()
	net.initialize(ctx=ctx,init=init.Xavier())
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)
	d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

**练习**
1.与AlexNet相比，VGG通常计算慢很多，也需要更多的内存或显存。试分析原因。
	
	与AlexNet相比，VGG的总层数增多了，模型复杂度增加了

**5.8 网络中的网络（NiN）**

LeNet、AlexNet和VGG在设计上的共同之处是：先以由卷积层构成的模块充分抽取空间特征，再以由全连接层构成的模块来输出分类结果。其中，AlexNet和VGG对LeNet的改进主要在于如何对这两个模块加宽（增加通道数）和加深。   
网络中的网络（NiN）重复使用由卷积层和代替全连接层的 1×1 卷积层构成的NiN块来构建深层网络。

**1. NiN块**

卷积层的输入和输出通常是四维数组（样本，通道，高，宽），而全连接层的输入和输出则通常是二维数组（样本，特征）。如果想在全连接层后再接上卷积层，则需要将全连接层的输出变换为四维。  
1×1 卷积层可以看成全连接层，其中空间维度（高和宽）上的每个元素相当于样本，通道相当于特征。因此，NiN使用 1×1 卷积层来替代全连接层，从而使空间信息能够自然传递到后面的层中去。

NiN块是NiN中的基础块。它由一个卷积层加两个充当全连接层的 1×1 卷积层串联而成。其中第一个卷积层的超参数可以自行设置，而第二和第三个卷积层的超参数一般是固定的。

	import d2lzh as d2l
	from mxnet import gluon,init,nd
	from mxnet.gluon import nn
	# 实现NiN块
	def nin_block(num_channels,kernel_size,strides,padding):
	    blk=nn.Sequential()
	    blk.add(nn.Conv2D(num_channels,kernel_size,strides,padding,activation='relu'),
	           nn.Conv2D(num_channels,kernel_size=1,activation='relu'),
	           nn.Conv2D(num_channels,kernel_size=1,activation='relu'))
	    return blk

**3. NiN模型**

NiN使用卷积窗口形状分别为11×11、5×5和3×3的卷积层，相应的输出通道数也与AlexNet中的一致。每个NiN块后接一个步幅为2、窗口形状为 3×3 的最大池化层。      
除使用NiN块以外，NiN还有一个设计与AlexNet显著不同：NiN去掉了AlexNet最后的3个全连接层，取而代之地，NiN使用了输出通道数等于标签类别数的NiN块，然后使用全局平均池化层（即窗口形状等于输入空间维形状的平均池化层）对每个通道中所有元素求平均并直接用于分类。NiN的这个设计的好处是可以显著减小模型参数尺寸，从而缓解过拟合。然而，该设计有时会造成获得有效模型的训练时间的增加。

	# 实现NiN模型
	net=nn.Sequential()
	net.add(nin_block(96,kernel_size=11,strides=4,padding=0),
	       nn.MaxPool2D(pool_size=3,strides=2),
	       nin_block(256,kernel_size=5,strides=1,padding=2),
	       nn.MaxPool2D(pool_size=3,strides=2),
	       nin_block(384,kernel_size=3,strides=1,padding=1),
	       nn.MaxPool2D(pool_size=3,strides=2),nn.Dropout(0.5),
	       # 标签类别数是10使用了输出通道数等于标签类别数的NiN块代替输出的全连接层 
	       nin_block(10,kernel_size=3,strides=1,padding=1), 
	       # 全局平均池化层将窗口形状自动设置成输入的高和宽,形状为(批量大小,10,1,1)
	       nn.GlobalAvgPool2D(),
	       # 将四维的输出转成二维的输出，其形状为(批量大小, 10) 
	       nn.Flatten() 
	       )

**4. 训练模型**

NiN的训练与AlexNet和VGG的类似，但这里使用的学习率更大。

	import mxnet as mx
	lr,num_epochs,batch_size,ctx=0.1,5,128,mx.cpu()
	net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=224)
	d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

**5.9 含并行连结的网络（GoogLeNet）**

**1. Inception 块**

GoogLeNet中的基础卷积块叫作Inception块，Inception块里有4条并行的线路。前3条线路使用窗口大小分别是 1×1 、 3×3 和 5×5 的卷积层来抽取不同空间尺寸下的信息，其中中间2个线路会对输入先做 1×1 卷积来减少输入通道数，以降低模型复杂度。第四条线路则使用 3×3 最大池化层，后接 1×1 卷积层来改变通道数。4条线路都使用了合适的填充来使输入与输出的高和宽一致。最后我们将每条线路的输出在通道维上连结，并输入接下来的层中去。

Inception块中可以自定义的超参数是每个层的输出通道数，我们以此来控制模型复杂度。

	import d2lzh as d2l
	from mxnet import gluon,init,nd
	from mxnet.gluon import nn
	
	class Inception(nn.Block):
	    # c1 - c4为每条线路里的层的输出通道数
	    # 输入通道数in_channels不指定时，将根据输入数据的形状进行推断in_channels。
	    def __init__(self,c1,c2,c3,c4,**kwargs):
	        super(Inception,self).__init__(**kwargs)
	        # 线路1，单1 x 1卷积层
	        self.p1_1=nn.Conv2D(c1,kernel_size=1,activation='relu')
	        # 线路2，1 x 1卷积层后接3 x 3卷积层
	        self.p2_1=nn.Conv2D(c2[0],kernel_size=1,activation='relu')
	        self.p2_2=nn.Conv2D(c2[1],kernel_size=3,padding=1,activation='relu')
	        # 线路3，1 x 1卷积层后接5 x 5卷积层
	        self.p3_1=nn.Conv2D(c3[0],kernel_size=1,activation='relu')
	        self.p3_2=nn.Conv2D(c3[1],kernel_size=5,padding=2,activation='relu')
	        # 线路4，3 x 3最大池化层后接1 x 1卷积层
	        self.p4_1=nn.MaxPool2D(pool_size=3,strides=1,padding=1)
	        self.p4_2=nn.Conv2D(c4,kernel_size=1,activation='relu')
	        
	    def forward(self,x):
	        p1=self.p1_1(x)
	        p2=self.p2_2(self.p2_1(x))
	        p3=self.p3_2(self.p3_1(x))
	        p4=self.p4_2(self.p4_1(x))
	        return nd.concat(p1,p2,p3,p4,dim=1)# 在通道维(dim=1)上连结输出


**2.GoogLeNet模型**

GoogLeNet跟VGG一样，在主体卷积部分中使用5个模块（block），每个模块之间使用步幅为2的 3×3 最大池化层来减小输出高宽。第一模块使用一个64通道的 7×7 卷积层。

	b1=nn.Sequential()
	b1.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3,activation='relu'),
	      nn.MaxPool2D(pool_size=3,strides=2,padding=1))
	
第二模块使用2个卷积层：首先是64通道的 1×1 卷积层，然后是将通道增大3倍的 3×3 卷积层。它对应Inception块中的第二条线路。

	b2=nn.Sequential()
	b2.add(nn.Conv2D(64,kernel_size=1,activation='relu'),
	      nn.Conv2D(192,kernel_size=3,padding=1,activation='relu'),
	      nn.MaxPool2D(pool_size=3,strides=2,padding=1))
	
第三模块串联2个完整的Inception块。

	b3=nn.Sequential()
	b3.add(Inception(64,(96,128),(16,32),32),
	       Inception(128,(128,192),(32,96),64),
	       nn.MaxPool2D(pool_size=3,strides=2,padding=1))
	
第四模块串联了5个Inception块。

	b4=nn.Sequential()
	b4.add(Inception(192, (96, 208), (16, 48), 64),
	       Inception(160, (112, 224), (24, 64), 64),
	       Inception(128, (128, 256), (24, 64), 64),
	       Inception(112, (144, 288), (32, 64), 64),
	       Inception(256, (160, 320), (32, 128), 128),
	       nn.MaxPool2D(pool_size=3,strides=2,padding=1))
	
第五模块有两个Inception块。第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均池化层来将每个通道的高和宽变成1。   
最后我们将输出变成二维数组后接上一个输出个数为标签类别数 的全连接层。

	b5=nn.Sequential()
	b5.add(Inception(256, (160, 320), (32, 128), 128),
	       Inception(384, (192, 384), (48, 128), 128),
	       nn.GlobalAvgPool2D())#全局平均池化层来将每个通道的高和宽变成1
	
	net=nn.Sequential()
	net.add(b1,b2,b3,b4,b5,nn.Dense(10))
	
	X=nd.random.uniform(shape=(1,1,96,96))
	net.initialize()
	for layer in net:
	    X=layer(X)
	    print(layer.name,'output shape:\t',X.shape)
	
**3. 获取数据和训练模型**

	import mxnet as mx
	lr,num_epochs,batch_size,ctx=0.1,5,128,mx.cpu()
	net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=96)
	d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

