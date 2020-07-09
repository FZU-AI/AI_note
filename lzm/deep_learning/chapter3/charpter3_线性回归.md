# 第三章 深度学习基础 #

在训练模型的时候，为了获取一个好的模型，都是通过最小化损失函数，为了使得损失函数在训练集上最小化，需要在训练集上进行迭代，然后计算出损失值的梯度，然后再利用梯度去更新权重参数，当网络的层数比较多时，我们就需要利用链式求导法则来计算梯度更新权重。

作为机器学习的一类，深度学习通常基于神经网络模型逐级表示越来越抽象的概念或模式。

单层神经网络：线性回归和softmax回归。 

多层神经网络：多层感知机(MLP)。

避免过拟合的常用方法：权重衰减和丢弃法(dropout)。

**3.1 线性回归**

**线性回归**的输出是一个连续值，适用于**回归问题**。

回归问题：预测房屋价格、气温、销售额等连续值的问题。

**1. 线性回归的基本要素**

**(1) 模型**

模型（model）：基于输入x<sub>1</sub>和x<sub>2</sub>来计算输出y的表达式。

线性回归假设输出与各个输入之间是线性关系：  
yˆ = x<sub>1</sub> w<sub>1</sub> + x<sub>2</sub> w<sub>2</sub> + b,   
其中w<sub>1</sub>和w<sub>2</sub>是权重（weight），b是偏差（bias），且均为标量。它们是线性回归模型的参数（parameter）。

输出 yˆ的计算依赖于x<sub>1</sub>和x<sub>2</sub>。也就是说，输出层中的神经元和输入层中各个输入完全连接。因此，这里的输出层又叫全连接层（fully-connected layer）或稠密层（dense layer）。 
          
模型输出 yˆ 是线性回归对真实价格y的预测或估计。

**(2) 模型训练**

模型训练（modeltraining）：通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小（即损失函数尽可能小）。

模型训练的3个要素：训练数据、损失函数、优化算法


**(3) 训练数据**

在预测房屋价格的问题中，一栋房屋被称为一个样本（sample），其真实售出价格叫作标签（label），用来预测标签的两个因素叫作特征 （feature）。特征用来表征样本的特点。

假设我们采集的样本数为n，索引为i的样本的特征为x<sub>1</sub><sup>(i)</sup>和x<sub>2</sub><sup>(i)</sup>，标签为y<sup>(i)</sup> , yˆ 是线性回归对真实价格y的预测或估计。对于索引为i的房屋，线性回归模型的房屋价格预测表达式为  
yˆ<sup>(i)</sup> = x<sub>1</sub><sup>(i)</sup> w<sub>1</sub> + x<sub>2</sub><sup>(i)</sup> w<sub>2</sub> + b

**(4) 损失函数**

损失函数（loss function）：衡量模型预测值与真实值之间的误差的函数。  
通常我们会选取一个非负数作为误差，且数值越小表示误差越小。  
索引为i的样本误差的损失函数的表达式为
ℓ<sup>(i)</sup>(w<sub>1</sub>,w<sub>2</sub>,b) = 1/2 * ( yˆ<sup>(i)</sup> −y<sup>(i)</sup> )<sup>2</sup> ，  
这里使用的平方误差函数也称为平方损失（square loss）。   
平均损失：用训练数据集中所有样本误差的平均来衡量模型预测的质量，即   
ℓ(w<sub>1</sub>,w<sub>2</sub>,b) = 1/n * ∑ ℓ<sup>(i)</sup>(w<sub>1</sub>,w<sub>2</sub>,b)  
在模型训练中，我们希望找出一组模型参数，记为w<sub>1</sub>*,w<sub>2</sub>*,b*,来使训练样本平均损失最小。

**(5) 优化算法**

解析解（analytical solution）：当模型和损失函数形式较为简单时，可以直接⽤公式表达出来的误差最小化问题的解。

数值解（numerical solution）：只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值的解。


超参数（hyperparameter）：每个小批量中的样本个数即批量大小|B|(batch size)和学习率η（learning rate），超参数是人为设定的，并不是通过模型训练出来的。

我们通常所说的“调参”指的正是调节超参数，例如通过反复试错来找到超参数合适的值。

**小批量随机梯度下降（mini-batch stochastic gradient descent）：** 

1.选取一组模型参数的初始值，如随机选取；

2.对参数进行多次迭代，使每次迭代都可能降低损失函数的值。在每次迭代中，先随机均匀采样一个由固定数目训练数据样本所组成的小批量（mini-batch）B，然后求小批量中数据样本的平均损失( ℓ(θ) )有关模型参数（θ）的导数（梯度grad），最后用此梯度与预先设定的一个正数(即η/|B|)的乘积作为模型参数在本次迭代的减小量。  
即θ ← θ − （η/|B|）* ∑∇<sub>θ</sub>ℓ<sup>(i)</sup>(θ), 其中(i为索引∈B)，模型参数θ = [w<sub>1</sub>,w<sub>2</sub>,b]<sup>⊤</sup>

**2. 线性回归的表示方法**

既可以用神经网络图表示线性回归，又可以用矢量计算表示该模型。

将两个向量**直接做矢量加法**比将这两个向量按元素逐一做标量加法的效率更高，应该尽可能采用矢量计算，以提升计算效率。

当数据样本数为n，特征数为d时，线性回归的矢量计算表达式为   
**yˆ** = **Xw** + b,  
其中模型输出 **yˆ** ∈R<sup>n×1</sup>，批量数据样本特征**X** ∈R<sup>n×d</sup>，权重**w** ∈R<sup>d×1</sup>，偏差b ∈R   
损失函数的矢量形式为ℓ(θ) =（1/2n）* ( yˆ−y)<sup>⊤</sup>(yˆ−y). 


**3.2 线性回归的从零开始实现**

本节介绍只利用NDArray和autograd来实现一个线性回归的训练。 
首先，导入本节中实验所需的包或模块。

	# matplotlib包可用于作图，且设置成嵌入(inline)显⽰。
    %matplotlib inline
	from IPython import display
	from matplotlib import pyplot as plt 
	from mxnet import autograd,nd
	import random
    
**1. 生成数据集**

	num_inputs= 2
	num_examples= 1000
	true_w = [2,-3.4]
	true_b = 4.2
	# 生成形状为(num_examples,num_inputs)的NDArray，其元素服从均值(loc)为0，标准差(scale)为1的正态分布(normal）
	features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
	#features[:,0]截取所有行，0列的元素，获得一个一维行向量,即1x1000
	labels = true_w[0] *features[:,0]+true_w[1] * features[:,1]+true_b 
	labels +=nd.random.normal(scale=0.01,shape=labels.shape)
	# features为1000x2,labels为1x1000

调用d2l.set_figsize()即可打印⽮量图并设置图的尺寸

	import d2lzh as d2l
	# 设置图的尺寸为（5，5），默认尺寸为（3.5，2.5） 
	d2l.set_figsize((5,5))
	# plt也在d2l包中，plot和scatter具体参照https://blog.csdn.net/Poul_henry/article/details/88602806
	# plot第一个参数为x轴，第二个参数为y轴，前两个为向量且长度相等，'o'为以点的方式连接,marksize为散点的大小
	plt.plot(features[:,1].asnumpy() , labels.asnumpy(),'o',markersize=1);# 加分号只显示图
	# scatter默认以'o'方式画图，s为散点的大小，s==marksize**2，即marksize=4与s=16的散点大小相同
	d2l.plt.scatter(features[:,1].asnumpy() , labels.asnumpy(),s=1);# 加分号只显示图

**2. 读取数据**

在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数： 它每次返回batch_size（批量大小）个随机样本的特征和标签。 

	# 本函数已保存在d2lzh包中方便以后使用 
	def data_iter(batch_size, features, labels):
	    num_examples =len(features)
	    indices =list(range(num_examples))
		# 将序列随机排列
	    random.shuffle(indices) 
		# range(0,num_examples,batch_size)为[0,num_examples)的序列，步长step为batch_size，step默认为1
	    for i in range(0,num_examples,batch_size):
	        # j代表从indices中按照batch_size大小提取的一系列随机排列的索引值
	        # min函数保证当样本不能被批量大小整除时，可以返回最后一批次的样本
	        j=nd.array(indices[i:min(i + batch_size,num_examples)])
			# yield创建一个可迭代的生成器
	        # take函数根据索引值j获得features和labels中的batch_size数量样本的数据集并返回对应元素
	        yield features.take(j),labels.take(j) 

**3. 初始化模型参数**	
	
	w = nd.random.normal(scale=0.01,shape=(num_inputs,1))
	b = nd.zeros(shape=(1,))
	# 对这些参数求梯度来迭代参数的值，因此我们需要创建它们的梯度
	w.attach_grad()
	b.attach_grad()
	
**4. 定义模型**
 
	# 本函数已保存在d2lzh包中方便以后使用 
	def linreg(X,w,b): # X为 10x2，w为2x1
	    return nd.dot(X,w)+b # return 10x1

**5. 定义损失函数**

	# 本函数已保存在d2lzh包中方便以后使用 
	# 模型预测值y_hat,真实值y
	def squared_loss(y_hat,y):
	# y_hat为10x1，y为1x10，故要reshape
	    return (y_hat-y.reshape(y_hat.shape))**2/2

**6. 定义优化算法**

这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值。 

	# 本函数已保存在d2lzh包中方便以后使用 
	# sgd函数实现了小批量随机梯度下降算法
	# lr为学习率，batch_size为批量大小|B|,(lr/|B|)*param.grad为模型参数在本次迭代的减小量
	def sgd(params,lr,batch_size):
	    for param in params:
	        param[:] =param -lr *param.grad /batch_size

**7. 训练模型**

在训练模型的时候，为了获取一个好的模型，都是通过最小化损失函数，为了使得损失函数在训练集上最小化，需要在训练集上进行迭代，然后计算出损失值的梯度，然后再利用梯度去更新权重参数，当网络的层数比较多时，我们就需要利用链式求导法则来计算梯度更新权重。

	#学习率和迭代周期个数均为超参数
	lr =0.03 # 学习率
	num_epochs = 3 #迭代周期个数
	net = linreg #模型,为预测值
	loss =squared_loss #损失函数
	
	for epoch in range(1,num_epochs+1):# 训练模型一共需要num_epochs个迭代周期 
	    # 在每一个迭代周期中，会使用训练数据集中所有样本预测一次（假设样本数能够被批量大小整除）。
	    # X和y分别是小批量样本的特征和标签,data_iter为每次返回batch_size大小的数据集
	    for X,y in data_iter(batch_size,features,labels):
	        with autograd.record():
	            l = loss(net(X,w,b),y)# l是⼩批量样本关于X和y的loss 
	        l.backward() # 对小批量样本的损失l求关于模型参数w,b的梯度
	        sgd([w,b],lr,batch_size) # 使用小批量随机梯度下降(sgd)算法迭代模型参数
	    train_l = loss(net(features, w, b), labels) #一个迭代周期中使用所有样本求得模型参数w，b后的所有样本的loss
		# mean()求所有数的平均值，并返回只有一个元素NDArray
	    print('epoch %d,loss %f' %(epoch,train_l.mean().asnumpy()))
	
**练习**

1.为什么squared_loss函数中需要使用reshape函数？

	答：模型预测值y_hat为10x1，真实值y为1x10,故对y用reshape为10x1

2.尝试使用不同的学习率，观察损失函数值的下降快慢

	学习率小时，损失函数下降较快，当在相同迭代次数下，损失函数值较大
		lr=0.01时
		epoch 1,loss 2.186219
		epoch 2,loss 0.287401
		epoch 3,loss 0.038027
		lr=0.03时
		epoch 1,loss 0.035091
		epoch 2,loss 0.000121
		epoch 3,loss 0.000049
		lr=0.05时
		epoch 1,loss 0.000049
		epoch 2,loss 0.000048
		epoch 3,loss 0.000049

3.如果样本个数不能被批量大小整除，data_iter函数的⾏为会有什么变化？

	答：当样本个数不能被批量大小整除时，min函数保证可以返回最后一批次的样本
	但是由于batch_size仍为原值(偏大,η/|B|偏小，θ ← θ − （η/|B|）* ∑∇（ℓ^(i)(θ).grad）)，
	导致最后一批次的样本的模型参数在sgd梯度下降函数中迭代后的模型参数θ偏大

**3.3 线性回归的简洁实现**

介绍如何使用MXNet提供的Gluon接口更方便地实现线性回归的训练。

**1. 生成数据集**
	
	from mxnet import autograd,nd
	
	num_inputs= 2
	num_examples= 1000
	true_w = [2,-3.4]
	true_b = 4.2
	features = nd.random.normal(scale=1,shape=(num_examples,num_inputs))
	# features[:,0]截取所有行，0列的元素，获得一个一维行向量,即1x1000
	labels = true_w[0] *features[:,0]+true_w[1] * features[:,1]+true_b 
	labels +=nd.random.normal(scale=0.01,shape=labels.shape)
	# features为1000x2,labels为1x1000 

**2. 读取数据**

	from mxnet.gluon import data as gdata
	
	batch_size = 10
	# 将训练数据的特征和标签组合在一起
	dataset = gdata.ArrayDataset(features,labels)
	# DataLoader实例从dataset中随机读取batc_size大小的数据样本，shuffle代表随机,data_iter为一个迭代器
	data_iter =gdata.DataLoader(dataset,batch_size,shuffle=True)

**3. 定义模型**

nn是netural network的缩写，nn模块中定义了大量神经网路的层。

	from mxnet.gluon import nn

模型变量net，它是一个Sequential实例。在Gluon中，Sequential实例可以看作是一个串联各个层的容器。在构造模型时，我们在该容器中依次添加层。  

	net = nn.Sequential()

线性回归的输出层又叫全连接层。在Gluon中，全连接层是一个Dense实例。我们定义该层输出个数为1。add函数向模型变量net中添加全连接层。

	# Dense的output = activation(dot(input, weight) + bias)
	net.add(nn.Dense(1))

在Gluon中我们无须指定每一层输入的形状，例如线性回归的输入个数。
当模型变量net得到数据时，例如后面执行net(X)时，模型将自动推断出每一层的输入个数。

**4. 初始化模型参数**

init是initializer的缩写形式,init模块提供了模型参数初始化的各种方法。

	from mxnet import init

init.Normal(sigma=0.01)指定权重参数w每个元素将在初始化时随机采样于均值为0、标准差为0.01的正态分布。偏差参数b默认会初始化为零。 net中有模型参数w，b。
	
	# net.initialize()会默认初始化参数为[-0.7,0.7]的范围
	net.initialize(init.Normal(sigma=0.01))

**5. 定义损失函数**

在Gluon中，loss模块定义了各种损失函数。

	from mxnet.gluon import loss as gloss
	loss =gloss.L2Loss() # 平方损失又称L2范数损失 

**6. 定义优化算法**

在导入Gluon后，我们创建一个Trainer实例，并指定学习率为0.03的小批量随机梯度下降（sgd）为优化算法。

	from mxnet import gluon

net.collect_params()可以获取net实例所有层的全部参数，sgd算法将迭代这些参数，学习率lr为0.03

	# Trainer用'sgd'算法优化参数集合,net.collect_params()收集所有参数
	trainer =gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.03})

**7. 训练模型**

	num_epochs =3 
	for epoch in range(1,num_epochs+1):
	    for X,y in data_iter:
	        with autograd.record():
	            l = loss(net(X),y)# 小批量样本的loss，net(X)为模型预测值，y为真实值
	        l.backward()# 对l求模型参数w,b的梯度
			# 在step函数中指明批量大小，从而对批量中样本梯度求平均，梯度将由1/batch_size标准化。
            # 如果使用l = loss(net(X), y).mean()手动归一化损失，则将step设置为1。
	        trainer.step(batch_size)# 模型参数用sgd算法进行迭代
	    l=loss(net(features),labels)# 所有样本的loss
	    print('epoch %d,loss %f' %(epoch,l.mean().asnumpy()))
	dense = net[0] # 从net获得输出层
	dense.weight.data(),dense.bias.data() #获得w，b

**练习**

1.如果将l = loss(net(X), y)替换成l = loss(net(X), y).mean()，我们需要将trainer.step(batch_size)相应地改成trainer.step(1)。这是为什么呢？ 

	因为在进行sgd算法迭代时，是对小批量样品中每个样品的损失函数的梯度求和后除以batch_size求平均值，
	故在改为l = loss(net(X), y).mean()后计算的是小批量样本的平均损失，需要在进行sgd算法迭代时，
	将trainer.step(batch_size)相应地改成trainer.step(1)，不需要除以batch_size求平均值

2.如何访问dense.weight的梯度？

	dense.weight.grad()



