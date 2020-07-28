**5.10 批量归一化（batch normalization）**

对深层神经网络来说，即使输入数据已做标准化，训练中模型参数的更新依然很容易造成靠近输出层输出的剧烈变化。这种计算数值的不稳定性通常令我们难以训练出有效的深度模型。批量归一化（batch  normalization）能让较深的神经网络的训练变得更加容易。 
  
在模型训练时，批量归一化利用小批量上的均值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。

**1.批量归一化层**

对全连接层和卷积层做批量归一化的方法稍有不同。

**对全连接层做批量归一化**

通常，我们将批量归一化层置于全连接层中的仿射变换和激活函数之间。
  
设全连接层的输入为**u**，权重参数和偏差参数分别为**W** 和 **b**，激活函数为ϕ。由仿射变换 **x**=**Wu**+**b**得到批量归一化层的输入 **x** 。设批量归一化的运算符为**BN**。那么，使用批量归一化的全连接层的输出为ϕ(**BN**(**x**))。
          
考虑一个由m个样本组成的小批量，仿射变换的输出为一个新的小批量 B={**x**<sup>(1)</sup>,…,**x**<sup>(m)</sup>} 。它们正是批量归一化层的输入。对于小批量B中任意样本 **x**<sup>(i)</sup> ∈R<sup>d</sup>,1 ≤ i ≤ m，批量归一化层的输出**y** <sup>(i)</sup> = **BN**(**x**<sup>(i)</sup>), 同样是d维向量。
 
批量归一化过程如下：  
首先，对小批量B求均值和方差：   
**µ**<sub>B</sub> ← 1/m ∑ **x**<sup>(i)</sup>，  
**σ**<sup>2</sup><sub>B</sub> ← 1/m ∑(**x**<sup>(i)</sup> − **µ**<sub>B</sub>)<sup>2</sup>，其中的平方计算是按元素求平方。

接下来，使用按元素开方和按元素除法对**x**<sup>(i)</sup>标准化：  
 **x**ˆ<sup>(i)</sup> ← ( **x**<sup>(i)</sup> − **µ** <sub>B</sub> ) / √(**σ**<sup>2</sup><sub>B</sub>  + ϵ)
，  这里ϵ > 0是一个很小的常数，保证分母大于0。

在上面标准化的基础上，批量归一化层引入了两个可以学习的模型参数，拉伸（scale）参数**γ**和偏移（shift）参数**β** 。这两个参数和**x**<sup>(i)</sup>形状相同，皆为d维向量。它们与**x**ˆ<sup>(i)</sup>分别做按元素乘法（符号⊙）和加法计算：  
 **y**<sup>(i)</sup> ← **γ** ⊙ **x**ˆ<sup>(i)</sup> + **β**。至此，我们得到了 **x**<sup>(i)</sup>的批量归一化的输出 **y**<sup>(i)</sup>。

值得注意的是，可学习的拉伸和偏移参数保留了不对**x**<sup>(i)</sup>做批量归一化的可能：此时只需学出**γ** =√(**σ**<sup>2</sup><sub>B</sub>  + ϵ)和**β** = **µ**<sub>B</sub>。我们可以对此这样理解：如果批量归一化无益，理论上，学出的模型可以不使用批量归一化。 

**对卷积层做批量归一化**

对卷积层来说，批量归一化发生在卷积计算之后、应用激活函数之前。

如果卷积计算输出多个通道，我们需要对这些通道的输出分别做批量归一化，且每个通道都拥有独立的拉伸和偏移参数，并均为标量。设小批量中有m个样本。在单个通道上，假设卷积计算输出的高和宽分别为p和q。我们需要对该通道中m×p×q个元素同时做批量归一化。对这些元素做标准化计算时，我们使用相同的均值和方差，即该通道中m×p×q个元素的均值和方差。

**预测时的批量归一化**

使用批量归一化训练时，我们可以将批量大小设得大一点，从而使批量内样本的均值和方差的计算都较为准确。将训练好的模型用于预测时，我们希望模型对于任意输入都有确定的输出。因此，单个样本的输出不应取决于批量归一化所需要的随机小批量中的均值和方差。   

一种常用的方法是通过移动平均估算整个训练数据集的样本均值和方差，并在预测时使用它们得到确定的输出。可见，和丢弃层一样，批量归一化层在训练模式和预测模式下的计算结果也是不一样的。

**2. 简洁实现**

Gluon中nn模块定义的BatchNorm类使用起来更加简单。它不需要指定自己定义的BatchNorm类中所需的num_features和num_dims参数值。在Gluon中，这些参数值都将通过延后初始化而自动获取。下面我们用Gluon实现使用批量归一化的LeNet。

	net = nn.Sequential()
	net.add(nn.Conv2D(6, kernel_size=5),
	        nn.BatchNorm(),
	        nn.Activation('sigmoid'),
	        nn.MaxPool2D(pool_size=2, strides=2),
	        nn.Conv2D(16, kernel_size=5),
	        nn.BatchNorm(),
	        nn.Activation('sigmoid'),
	        nn.MaxPool2D(pool_size=2, strides=2),
	        nn.Dense(120),
	        nn.BatchNorm(),
	        nn.Activation('sigmoid'),
	        nn.Dense(84),
	        nn.BatchNorm(),
	        nn.Activation('sigmoid'),
	        nn.Dense(10))
	# 训练模型
	lr, num_epochs, batch_size, ctx = 1.0, 5, 256, d2l.try_gpu()
	net.initialize(ctx=ctx, init=init.Xavier())
	trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
	train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
	d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

**5.11 残差网络（ResNet）**

对神经网络模型添加新的层，充分训练后的模型训练误差仍可能升高。理论上，原模型解的空间只是新模型解的空间的子空间。也就是说，如果我们能将新添加的层训练成恒等映射 f(x)=x ，新模型和原模型将同样有效。由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差。然而在实践中，添加过多的层后训练误差往往不降反升。即使利用批量归一化带来的数值稳定性使训练深层模型更加容易，该问题仍然存在。针对这一问题，残差网络（ResNet）应运而生。

**1. 残差块**

设输入为x，经过由加权运算1（卷积+批量归一化）、激活函数、加权运算2（卷积+批量归一化），拟合出有关恒等映射的残差映射f(x)−x。再将残差映射f(x)−x与输入x相加得到f(x)，最后将激活函数作用于f(x)。以上构成了ResNet的基础块，即残差块（residual block）。   

以恒等映射作为我们希望学出的理想映射f(x)，若将加权运算2的权重和偏差参数学成0，那么有f(x)−x=0，即f(x)=x为恒等映射。实际中，当理想映射f(x)极接近于恒等映射时，残差映射也易于捕捉恒等映射的细微波动。在残差块中，输入可通过跨层的数据线路更快地向前传播，从而能够训练出有效的深度神经网络。

ResNet沿用了VGG全 3×3 卷积层的设计。残差块里首先有2个有相同输出通道数的 3×3 卷积层。每个卷积层后接一个批量归一化层和ReLU激活函数。然后我们将输入跳过这2个卷积运算后直接加在最后的ReLU激活函数前。这样的设计要求2个卷积层的输出与输入形状一样，从而可以相加。如果想改变通道数，就需要引入一个额外的 1×1 卷积层来将输入x变换成需要的形状后再做相加运算。

残差块的实现如下。它可以设定输出通道数、是否使用额外的 1×1 卷积层来修改通道数以及卷积层的步幅。

	import d2lzh as d2l
	from mxnet import gluon,init,nd
	from mxnet.gluon import nn
	
	# 本类已保存在d2lzh包中方便以后使用
	class Residual(nn.Block):
	    def __init__(self,num_channels,use_1x1conv=False,strides=1,**kwargs):
	        super(Residual,self).__init__(**kwargs)
	        self.conv1=nn.Conv2D(num_channels,kernel_size=3,padding=1,strides=strides)# 两个3x3卷积层
	        self.conv2=nn.Conv2D(num_channels,kernel_size=3,padding=1)
	        if use_1x1conv:#是否使用1x1卷积层来修改通道数以及卷积层的步幅。
	            self.conv3=nn.Conv2D(num_channels,kernel_size=1,strides=strides)
	        else:
	            self.conv3=None
	        self.bn1=nn.BatchNorm()# 两个批量归一化层
	        self.bn2=nn.BatchNorm()
	            
	    def forward(self,X):
	        Y=nd.relu(self.bn1(self.conv1(X)))# 加权运算1(卷积和批量归一化)和激活函数
	        Y=self.bn2(self.conv2(Y))# 加权运算2(卷积和批量归一化)，得到f(x)-x
	        if self.conv3:
	            X=self.conv3(X)
	        return nd.relu(Y+X)# Y+X即f(x),在应用激活函数

	# 在增加输出通道数的同时减半输出的高和宽。
	blk=Residual(6,use_1x1conv=True,strides=2)

**2. ResNet模型**

ResNet的前两层跟之前介绍的GoogLeNet中的一样：在输出通道数为64、步幅为2的 7×7 卷积层后接步幅为2的 3×3 的最大池化层。不同之处在于ResNet每个卷积层后增加的批量归一化层。

	net=nn.Sequential()
	net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
	       nn.BatchNorm(),nn.Activation('relu'),# 批量归一化发生在卷积计算之后、应用激活函数之前。
	       nn.MaxPool2D(pool_size=3,strides=2,padding=1)
	       )
	
ResNet则使用4个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。第一个模块的通道数同输入通道数一致。由于之前已经使用了步幅为2的最大池化层，所以无须减小高和宽。之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。

下面我们来实现这个模块。注意，这里对第一个模块做了特别处理。

	def resnet_block(num_channels,num_residuals,first_block=False):
	    blk=nn.Sequential()
	    for i in range(num_residuals):# 残差块数
	        if i==0 and not first_block:# 除第一个残差模块的第一个残差块将上一个模块的通道数翻倍，并将高和宽减半。
	            blk.add(Residual(num_channels,use_1x1conv=True,strides=2))
	        else:
	            blk.add(Residual(num_channels))
	    return blk
	
每个模块使用2个残差块。

	net.add(resnet_block(64,2,first_block=True),# 第一个残差模块不用减半
	       resnet_block(128,2),
	       resnet_block(256,2),
	       resnet_block(512,2))
	
最后，与GoogLeNet一样，加入全局平均池化层后接上全连接层输出。

	net.add(nn.GlobalAvgPool2D(),nn.Dense(10))

这里每个模块里有4个卷积层（不计算 1×1 卷积层），加上最开始的卷积层和最后的全连接层，共计18层。这个模型通常也被称为ResNet-18。

**5.11.3 训练模型**

	import mxnet as mx
	lr,num_epochs,batch_size,ctx=0.05,5,256,mx.cpu()
	net.initialize(force_reinit=True,ctx=ctx,init=init.Xavier())
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=96)
	d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)

**5.12 稠密连接网络（DenseNet）**

将部分前后相邻的运算抽象为模块A（输入X）和模块B（输出Y），DenseNet与ResNet的主要区别在于，DenseNet里模块B的输出不是像ResNet那样和模块A的输出相加，而是在通道维上连结。这样模块A的输出可以直接传入模块B后面的层。在这个设计里，模块A直接跟模块B后面的所有层连接在了一起。这也是它被称为“稠密连接”的原因。

在跨层连接上，不同于ResNet中将输入与输出相加，DenseNet在通道维上连结输入与输出。   
DenseNet的主要构建模块是稠密块（dense block）和过渡层（transition layer）。前者定义了输入和输出是如何连结的，后者则用1x1卷积层来控制通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。

**1. 稠密块**

DenseNet使用了ResNet改良版的“批量归一化、激活和卷积”的卷积块结构，我们首先在`conv_block`函数里实现这个卷积块结构。

	import d2lzh as d2l
	from mxnet import gluon,init,nd
	from mxnet.gluon import nn
	
	def conv_block(num_channels):# num_channels为输出通道数
	    blk=nn.Sequential()
	    blk.add(nn.BatchNorm(),nn.Activation('relu'),# “批量归一化、激活和卷积”结构 
	           nn.Conv2D(num_channels,kernel_size=3,padding=1))
	    return blk
	
稠密块由num_convs个conv_block组成，每个卷积块使用相同的输出通道数。但在前向计算时，我们将每个卷积块的输入X和输出Y在通道维上连结。

	class DenseBlock(nn.Block):
	    def __init__(self,num_convs,num_channels,**kwargs):
	        super(DenseBlock,self).__init__(**kwargs)
	        self.net=nn.Sequential()
	        for _ in range(num_convs):# num_convs为卷积块数
	            self.net.add(conv_block(num_channels))# num_channels为输出通道数
	            
	    def forward(self,X):
	        for blk in self.net:
	            Y=blk(X)
	            X=nd.concat(X,Y,dim=1)# 在通道维上将输入和输出连结
	        return X
	
定义一个有n个输出通道数为m的卷积块。使用通道数为t的输入时，我们会得到通道数为 t+n×m 的输出。卷积块的通道数控制了输出通道数相对于输入通道数的增长，因此也被称为增长率（growth rate）。
	
**2. 过渡层**

由于每个稠密块都会带来通道数的增加，使用过多则会带来过于复杂的模型。过渡层用来控制模型复杂度。它通过 1×1 卷积层来减小通道数，并使用步幅为2的平均池化层减半高和宽，从而进一步降低模型复杂度。过渡层也使用“批量归一化、激活和卷积”的卷积块结构

	def transition_block(num_channels):# 输出通道数为num_channels
	    blk=nn.Sequential()
	    blk.add(nn.BatchNorm(),nn.Activation('relu'),
	           nn.Conv2D(num_channels,kernel_size=1),
	           nn.AvgPool2D(pool_size=2,strides=2))
	    return blk
	
**3. DenseNet模型**

DenseNet首先使用同ResNet一样的单卷积层和最大池化层。

	net=nn.Sequential()
	net.add(nn.Conv2D(64,kernel_size=7,strides=2,padding=3),
	       nn.BatchNorm(),nn.Activation('relu'),
	       nn.MaxPool2D(pool_size=3,strides=2,padding=1))
	
DenseNet使用4个稠密块。可以设置每个稠密块使用多少个卷积层。稠密块里的卷积层通道数（即增长率）设为32，所以每个稠密块将增加128个通道。   
ResNet里通过步幅为2的残差块在每个模块之间减小高和宽。这里我们则使用过渡层来减半高和宽，并减半通道数。

	num_channels,growth_rate=64,32 # num_channels为当前的通道数
	num_convs_in_dense_blocks=[4,4,4,4]# 每个稠密块的卷积层数。
	
	for i,num_convs in enumerate(num_convs_in_dense_blocks):# enumerate获得(index,value)的迭代器
	    net.add(DenseBlock(num_convs,growth_rate))
	    # 上一个稠密块的输出通道数
	    num_channels+=num_convs*growth_rate
	    # 在稠密块之间加入通道数减半的过渡层
	    if i !=len(num_convs_in_dense_blocks)-1:# 不是最后一个稠密块，则加入过渡层
	        num_channels//=2
	        net.add(transition_block(num_channels))        
	
同ResNet一样，最后接上全局池化层和全连接层来输出。

	net.add(nn.BatchNorm(),nn.Activation('relu'),nn.GlobalAvgPool2D(),
	       nn.Dense(10))
	
**4. 获取数据并训练模型** 

	import mxnet as mx
	lr,num_epochs,batch_size,ctx=0.1,5,256,mx.cpu()
	net.initialize(ctx=ctx,init=init.Xavier())
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size,resize=96)
	d2l.train_ch5(net,train_iter,test_iter,batch_size,trainer,ctx,num_epochs)
	
