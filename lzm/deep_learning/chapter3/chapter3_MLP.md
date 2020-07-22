**3.8 多层感知机（multilayer perceptron,MLP）**

**多层感知机在输出层与输入层之间加入了一个或多个全连接隐藏层，并通过激活函数对隐藏层输出进行变换。**

**1. 隐藏层** 

多层感知机在单层神经网络的基础上引入了一到多个隐藏层（hiddenlayer）。隐藏层位于输入层和输出层之间。多层感知机中的隐藏层和输出层都是**全连接层**。

给定一个小批量样本**X** ∈R<sup>nxd</sup>，其批量大小为n，输入个数为d。假设多层感知机只有一个隐藏层，其中隐藏单元个数为h。记隐藏层的输出（也称为隐藏层变量或隐藏变量）为**H**， 有**H** ∈ R<sup>n×h</sup>。因为隐藏层和输出层均是全连接层，可以设隐藏层的权重参数和偏差参数分别 为**W**<sub>h</sub> ∈R<sup>d×h</sup>和**b** <sub>h</sub> ∈R<sup>1×h</sup>，输出层的权重和偏差参数分别为**W**<sub>o</sub> ∈R<sup>h×q</sup>和 **b**<sub>o</sub> ∈R<sup>1×q</sup>。 其输出 **O** ∈R<sup>n×q</sup>的计算为   
**H** = **XW**<sub>h</sub> + **b**<sub>h</sub>,  
**O** = **HW**<sub>o</sub> + **b**<sub>o</sub>,  

也就是将**隐藏层的输出直接作为输出层的输入**。  

如果将以上两个式子联立起来，可以得到   
**O** = (**XW**<sub>h</sub> +**b**<sub>h</sub>) **W**<sub>o</sub> + **b**<sub>o</sub> = **XW**<sub>h</sub>**W**<sub>o</sub> + **b**<sub>h</sub>**W**<sub>o</sub> + **b**<sub>o</sub>. 

从联立后的式子可以看出，虽然神经网络引入了隐藏层，却依然等价于一个单层神经网络：其中输出层权重参数为 **W**<sub>h</sub>**W**<sub>o</sub>，偏差参数为 **b**<sub>h</sub>**W**<sub>o</sub> + **b**<sub>o</sub>。不难发现，即便再添加更多的隐藏层，以上设计依然只能与仅含输出层的单层神经网络等价。

**2. 激活函数（activation function）**  

上述问题的根源在于全连接层只是对数据做仿射变换（affine transformation），而多个仿射变换 的叠加仍然是一个仿射变换。解决问题的一个方法是**引入非线性变换**，例如对隐藏变量使用按元素运算的非线性函数进行变换，然后再作为下一个全连接层的输入。这个非线性函数被称为激活函数（activation function）。

常用的激活函数：  
(1)ReLU函数：ReLU(x) = max(x,0)  
ReLU函数只保留正数元素，并将负数元素清零。

(2)sigmoid函数：sigmoid(x) = 1/(1 +exp(−x))   
sigmoid函数可以将元素的值变换到0和1之间。

(3)tanh函数：tanh(x) = (1−exp(−2x))/(1 +exp(−2x))  
tanh（双曲正切）函数可以将元素的值变换到-1和1之间。

**3. 多层感知机** 

多层感知机就是含有至少一个隐藏层的由全连接层组成的神经网络，且每个隐藏层的输出通过激活函数进行变换。多层感知机的层数和各隐藏层中隐藏单元个数都是超参数。多层感知机按以下方式计算输出：  
**H** = ϕ(**XW**<sub>h</sub> + **b**<sub>h</sub>）,  
**O** = **HW**<sub>o</sub> + **b**<sub>o</sub>,  
ϕ表示激活函数。  

在分类问题中，我们可以对输出**O**做softmax运算，并使⽤softmax回归中 的交叉熵损失函数。在回归问题中，我们将输出层的输出个数设为1，并将输出**O**直接提供给线性回归中使用的平方损失函数。 

**3.9 多层感知机的从零开始实现** 

	# 1.导入所需的包或模块
	%matplotlib inline
	import d2lzh as d2l
	from mxnet import nd
	from mxnet.gluon import loss as gloss
	import time
	
	# 2. 获取和读取数据 
	batch_size=256
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
	
	# 3. 定义模型参数
	# 输入个数，输出个数，隐藏单元个数
	num_inputs,num_outputs,num_hiddens =784,10,256
	
	W1=nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens))
	b1=nd.zeros(num_hiddens)
	W2=nd.random.normal(scale=0.01,shape=(num_hiddens,num_outputs))
	b2=nd.zeros(num_outputs)
	params =[W1,b1,W2,b2]
	# 为w,b申请梯度的内存
	for param in params:
	    param.attach_grad()
	
	# 4. 定义激活函数
	def relu(X):
	    return nd.maximum(X,0)
	
	# 5. 定义模型
	def net(X):
	    X=X.reshape((-1,num_inputs))# X.reshape为 样本数x784
	    H=relu(nd.dot(X,W1)+b1)# W1为784x256
	    return nd.dot(H,W2)+b2# W2为256x10
	
	# 6. 定义损失函数 
	# 为了得到更好的数值稳定性，我们直接使用Gluon提供的包括softmax运算和交叉熵损失计算的函数。 
	loss=gloss.SoftmaxCrossEntropyLoss()
	
	# 7. 训练模型 
	start=time.time()
	num_epochs,lr =5,0.5
	d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
	time=time.time()-start


**练习**

练习1.改变超参数num_hiddens的值，看看对实验结果有什么影响。 

	随着隐藏单元(num_hiddens)的增加，训练时间增加
	
	#128
	epoch 1, loss 0.8946, train acc 0.664, test acc 0.810
	epoch 2, loss 0.5108, train acc 0.810, test acc 0.844
	epoch 3, loss 0.4461, train acc 0.835, test acc 0.857
	epoch 4, loss 0.4060, train acc 0.852, test acc 0.863
	epoch 5, loss 0.3924, train acc 0.856, test acc 0.864
	time
	34.10546922683716

	# 256
	epoch 1, loss 0.7824, train acc 0.706, test acc 0.818
	epoch 2, loss 0.4825, train acc 0.822, test acc 0.853
	epoch 3, loss 0.4271, train acc 0.842, test acc 0.858
	epoch 4, loss 0.3922, train acc 0.854, test acc 0.866
	epoch 5, loss 0.3708, train acc 0.863, test acc 0.876
	time
	38.06653618812561

	# 512
	epoch 1, loss 0.8166, train acc 0.698, test acc 0.819
	epoch 2, loss 0.4849, train acc 0.818, test acc 0.816
	epoch 3, loss 0.4278, train acc 0.842, test acc 0.861
	epoch 4, loss 0.3896, train acc 0.855, test acc 0.866
	epoch 5, loss 0.3672, train acc 0.863, test acc 0.872
	time
	45.47329878807068

  	# 1024
	epoch 1, loss 0.7732, train acc 0.713, test acc 0.822
	epoch 2, loss 0.4768, train acc 0.823, test acc 0.852
	epoch 3, loss 0.4232, train acc 0.844, test acc 0.864
	epoch 4, loss 0.3817, train acc 0.859, test acc 0.866
	epoch 5, loss 0.3679, train acc 0.864, test acc 0.873
	time
	60.22583317756653

练习2. 试着加入一个新的隐藏层，看看对实验结果有什么影响。 

	训练时间增加了，预测的准确率下降了

	# 加入一个隐藏层修改的代码如下
	# 输入个数，输出个数，隐藏单元个数
	num_inputs,num_outputs,num_hiddens1,num_hiddens2 =784,10,256,256
	
	W1=nd.random.normal(scale=0.01,shape=(num_inputs,num_hiddens1))
	b1=nd.zeros(num_hiddens1)
	W2=nd.random.normal(scale=0.01,shape=(num_hiddens1,num_hiddens2))
	b2=nd.zeros(num_hiddens2)
	W3=nd.random.normal(scale=0.01,shape=(num_hiddens2,num_outputs))
	b3=nd.zeros(num_outputs)
	params =[W1,b1,W2,b2,W3,b3]
	# w,b申请梯度的内存
	for param in params:
	    param.attach_grad()
	
	def net(X):
	    X=X.reshape((-1,num_inputs))# X.reshape为 样本数x784
	    H1=relu(nd.dot(X,W1)+b1)# W1为784x256
	    H2=relu(nd.dot(H1,W2)+b2)
	    return nd.dot(H2,W3)+b3# W2为256x10
	
	start=time.time()
	num_epochs,lr =5,0.5
	d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)
	time=time.time()-start
	time
	----
	epoch 1, loss 1.1145, train acc 0.564, test acc 0.787
	epoch 2, loss 0.7082, train acc 0.746, test acc 0.824
	epoch 3, loss 0.4766, train acc 0.823, test acc 0.845
	epoch 4, loss 0.4316, train acc 0.841, test acc 0.859
	epoch 5, loss 0.3996, train acc 0.850, test acc 0.854
	39.58924889564514

**3.10 多层感知机的简洁实现** 

	# 1.导入所需的包或模块
	import d2lzh as d2l
	from mxnet import gluon,init
	from mxnet.gluon import loss as gloss,nn
	
	# 2. 定义模型
	net=nn.Sequential()
	# 隐藏层单元个数为256， 并使用ReLU函数作为激活函数。
	net.add(nn.Dense(256,activation='relu'),
	        nn.Dense(10))
	net.initialize(init.Normal(sigma=0.01))
	
	# 3. 读取数据并训练模型 
	batch_size =256
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
	
	loss=gloss.SoftmaxCrossEntropyLoss()
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.5})
	num_epochs=5
	start=time.time()
	d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)
	time=time.time()-start

**练习**

1.尝试多加入几个隐藏层，对比上一节中从零开始的实现
	
	随着隐藏层的层数的增加，模型预测的准确率下降，训练的时间增长了
 
	net.add(nn.Dense(256,activation='relu'),
        nn.Dense(10))
	1
	epoch 1, loss 0.7982, train acc 0.705, test acc 0.829
	epoch 2, loss 0.4886, train acc 0.820, test acc 0.852
	epoch 3, loss 0.4270, train acc 0.842, test acc 0.856
	epoch 4, loss 0.3953, train acc 0.855, test acc 0.866
	epoch 5, loss 0.3689, train acc 0.863, test acc 0.871
	time
	36.296797037124634
	
	net.add(nn.Dense(256,activation='relu'),nn.Dense(256,activation='relu'),
        nn.Dense(10))
	2
	epoch 1, loss 1.1123, train acc 0.562, test acc 0.780
	epoch 2, loss 0.5751, train acc 0.786, test acc 0.836
	epoch 3, loss 0.4556, train acc 0.832, test acc 0.850
	epoch 4, loss 0.5764, train acc 0.812, test acc 0.782
	epoch 5, loss 0.5038, train acc 0.817, test acc 0.860
	time
	38.1756477355957
	
	net.add(nn.Dense(256,activation='relu'),nn.Dense(256,activation='relu'),nn.Dense(256,activation='relu'),
        nn.Dense(10))
	3
	epoch 1, loss 1.8864, train acc 0.247, test acc 0.538
	epoch 2, loss 0.8791, train acc 0.642, test acc 0.742
	epoch 3, loss 0.5930, train acc 0.770, test acc 0.811
	epoch 4, loss 0.6992, train acc 0.752, test acc 0.828
	epoch 5, loss 0.4749, train acc 0.823, test acc 0.850
	time
	43.10390615463257

2.使用其他的激活函数，看看对结果的影响

	使用sgimoid和tanh函数的模型预测率均没relu好
	
	net.add(nn.Dense(256,activation='sigmoid'),
	        nn.Dense(10))
	epoch 1, loss 1.0739, train acc 0.608, test acc 0.769
	epoch 2, loss 0.5789, train acc 0.785, test acc 0.805
	epoch 3, loss 0.5062, train acc 0.815, test acc 0.830
	epoch 4, loss 0.4713, train acc 0.828, test acc 0.840
	epoch 5, loss 0.4446, train acc 0.837, test acc 0.850
	time
	34.85145449638367

	net.add(nn.Dense(256,activation='tanh'),
	        nn.Dense(10))
	epoch 1, loss 0.7753, train acc 0.714, test acc 0.828
	epoch 2, loss 0.5181, train acc 0.810, test acc 0.845
	epoch 3, loss 0.4640, train acc 0.830, test acc 0.849
	epoch 4, loss 0.4358, train acc 0.841, test acc 0.852
	epoch 5, loss 0.4074, train acc 0.851, test acc 0.853
	time
	36.659870862960815