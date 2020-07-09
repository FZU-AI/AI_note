**3.4 softmax回归**

**softmax回归**的输出是一个离散值，适用于**分类问题**。

分类问题：图像分类、垃圾邮件识别、疾病检测等输出为离散值的问题。

**(1) 模型**

softmax回归跟线性回归一样将输入特征与权重做线性叠加。softmax回归的输出值个数等于标签里的类别数。

假设每个样本有4个特征x<sub>1</sub>，x<sub>2</sub>，x<sub>3</sub>，x<sub>4</sub>；3个输出类别o<sub>1</sub>，o<sub>2</sub>，o<sub>3</sub>。需要计算每个样本的3个输出值，故需要12个权值w，3个偏差b。

o<sub>1</sub> = x<sub>1</sub>w<sub>11</sub> + x<sub>2</sub>w<sub>21</sub> + x<sub>3</sub>w<sub>31</sub> + x<sub>4</sub>w<sub>41</sub> + b<sub>1</sub>  
o<sub>2</sub> = x<sub>1</sub>w<sub>12</sub> + x<sub>2</sub>w<sub>22</sub> + x<sub>3</sub>w<sub>32</sub> + x<sub>4</sub>w<sub>42</sub> + b<sub>2</sub>  
o<sub>3</sub> = x<sub>1</sub>w<sub>13</sub> + x<sub>2</sub>w<sub>23</sub> + x<sub>3</sub>w<sub>33</sub> + x<sub>4</sub>w<sub>43</sub> + b<sub>3</sub>

由于每个样本的输出o<sub>1</sub>，o<sub>2</sub>，o<sub>3</sub>的计算都要依赖于所有的输入x<sub>1</sub>，x<sub>2</sub>，x<sub>3</sub>，x<sub>4</sub>，所以softmax回归的输出层也是一个全连接层。

**(2) softmax运算**

softmax运算不改变预测类别输出。 
softmax运算符（softmax operator）通过下式将输出值变换成值为正且和为1的概率分布：  
yˆ<sub>1</sub>，yˆ<sub>2</sub>，yˆ<sub>3</sub> = softmax(o<sub>1</sub>，o<sub>2</sub>，o<sub>3</sub>)，  
其中yˆ<sub>i</sub> =exp(o<sub>i</sub>)/∑ exp(o<sub>i</sub>)，i为1，2，3。

**(3) 单样本分类的矢量计算表达式**

softmax回归对样本i分类的矢量计算表达式为  
**o**<sup>(i)</sup> = **x**<sup>(i)</sup>**W** + **b**  

**y**ˆ<sup>(i)</sup> = softmax(**o**<sup>(i)</sup>)

**(4) 小批量样本分类的矢量计算表达式**

给定一个小批量样本，其批量大小为n，输入个数（特征数）为d，输出个数（类别数）为q。设批量特征为**X** ∈ R<sup>n×d</sup>。 假设softmax回归的权重和偏差参数分别为**W** ∈R<sup>d×q</sup>和**b** ∈R<sup>1×q</sup>。softmax回归的矢量计算表达式为  
 **O** = **XW**+ **b**，  
 **Yˆ**= softmax(**O**)，  

其中的加法运算使用了广播机制，**O**，**Yˆ**∈R<sup>n×q</sup>且这两个矩阵的第i行分别为样本i的输出 **o**<sup>(i)</sup>和概率分布 **yˆ**<sup>(i)</sup>。 

**(5) 交叉熵损失函数**

交叉熵（cross entropy）：
H(y<sup>(i)</sup>，yˆ<sup>(i)</sup>)= − ∑ 
y<sub>j</sub><sup>(i)</sup> log y<sub>j</sub><sup>ˆ(i)</sup>，其中带下标的y<sub>j</sub><sup>(i)</sup>是向量y<sup>(i)</sup>中非0即1的元素。

交叉熵适合衡量两个概率分布差异，只关心对正确类别的预测概率，因为只要其值足够大，就可以确保分类结果正确。

假设训练数据集的样本数为n，交叉熵损失函数定义为
ℓ(Θ) =1/n ∑ H(y<sup>(i)</sup>，yˆ<sup>(i)</sup>)

最小化交叉熵损失函数等价于最大化训练数据集所有标签类别的联合预测概率。

**3.5 图像分类数据集（Fashion-MNIST）**

**3.5.1 获取数据集**

首先导入本节需要的包或模块。 

	%matplotlib inline 
	import d2lzh as d2l 
	from mxnet.gluon import data as gdata 
	import sys 
	import time

通过参数train来指定获取训练数据集或测试数据集

	mnist_train = gdata.vision.FashionMNIST(train=True) # 训练集
	mnist_test = gdata.vision.FashionMNIST(train=False) # 测试集

通过方括号[]来访问任意一个样本，下面获取第一个样本的图像和标签。 

	feature, label = mnist_train[0] 

变量feature的形状为（28，28，1），对应高和宽均为28像素的图像。每个像素的数值为0到255之间8位无符号整数（uint8）。它使用三维的NDArray存储。其中的最后一维是通道数。因为数据集中是灰度图像，所以通道数为1。

图像的标签label使用NumPy的标量表示。它的类型为32位整数（int32）。

下面定义一个可以在一行里画出多张图像和对应标签的函数 。

	# 本函数已保存在d2lzh包中方便以后使用 
	def show_fashion_mnist(images, labels): 
	    d2l.use_svg_display() 
	    # 这里的_表示我们忽略（不使用）的变量
	    # 创建1*len(images)个子图即1行len(images)列，每个子图大小为(12, 12)
	    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12)) 
	    for f, img, lbl in zip(figs, images, labels): 
	        # reshape设置为(28,28)像素，并显示
	        f.imshow(img.reshape((28, 28)).asnumpy()) 
	        f.set_title(lbl) # 设置标题为lbl
	        # False设置不显示x，y轴
	        f.axes.get_xaxis().set_visible(True)
	        f.axes.get_yaxis().set_visible(True) 

**3.5.2 读取小批量**
	
	batch_size =256
	# ToTensor实例将图像数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值均在0到1之间。
	# ToTensor实例还将图像通道从最后一维移到最前一维来方便之后介绍的卷积神经网络计算。
	transformer =gdata.vision.transforms.ToTensor()
	# sys.platform获得操作系统的型号，startwith('win')判断字符串是否以win开始
	if sys.platform.startswith('win'):
	    num_workers =0 # 0表示不用额外的进程来加速读取数据， windows系统不支持使用多进程加速数据读取
	else:
	    num_workers =4
	# DataLoader实例每次读取一个样本数为batch_size的小批量数据。
	# DataLoader允许使用多进程来加速数据读取（暂不支持Windows操作系统），我们通过参数num_workers来设置4个进程读取数据。
	# 通过数据集的transform_first函数，我们将ToTensor实例的变换应用在每个数据样本（图像和标签）的第一个元素，即图像之上。 
	train_iter =gdata.DataLoader(mnist_train.transform_first(transformer),batch_size,shuffle=True,num_workers=num_workers)
	test_iter =gdata.DataLoader(mnist_test.transform_first(transformer),batch_size,shuffle=False,num_workers=num_workers)

我们将获取并读取Fashion-MNIST数据集的逻辑封装在`d2lzh. load_data_fashion_mnist`函数中供后面章节调用。

该函数将返回`train_iter`和`test_iter`两个变量。

**3.6 softmax回归的从零开始实现**

首先导入所需的包或模块

	%matplotlib inline
	import d2lzh as d2l
	from mxnet import autograd,nd

**3.6.1 获取和读取数据**

获取并读取Fashion-MNIST数据集，批量大小为256

	batch_size = 256
	train_iter,test_iter =d2l.load_data_fashion_mnist(batch_size)

**3.6.2初始化模型参数**

	num_inputs = 784 #输入为28x28像素的图片  样本数x784
	num_outputs =10 #输出个数
	
	W = nd.random.normal(scale=0.01,shape=(num_inputs,num_outputs))# W为784x10
	b = nd.zeros(num_outputs) # b为1x10
	
	# 为模型参数附上梯度
	W.attach_grad()
	b.attach_grad()

**3.6.3 实现softmax运算**

	X = nd.array([[1,2,3],[4,5,6]])
	# 只对其中同⼀列（axis=0）或同⼀行（axis=1）的元素求和，并在结果中保留行和列这两个维度（keepdims=True）
	X.sum(axis=0,keepdims=True),X.sum(axis=1,keepdims=True)

	out:
		[[5. 7. 9.]]
		 <NDArray 1x3 @cpu(0)>,
		 
		 [[ 6.]
		  [15.]]
		 <NDArray 2x1 @cpu(0)>)
	# keepdims=False默认保存为一维行向量
	X.sum(axis=0,keepdims=False),X.sum(axis=1,keepdims=False)

	out:
		[5. 7. 9.]
		 <NDArray 3 @cpu(0)>,
		 
		 [ 6. 15.]
		 <NDArray 2 @cpu(0)>)	

softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。 

	def softmax(X): # X的行数为样本数n，列数为输出个数q  X即输出o
	    X_exp =X.exp()# axis=1对同一行进行求和，并在结果中保留列这个维度(keepdims=True)
	    partition = X_exp.sum(axis=1,keepdims=True)
	    return X_exp/partition  # 应用了广播机制，X_exp为nxq ，partition为nx1


**3.6.4 定义模型**

	def net(X): #X.reshape后为256*784(256为样本数)  W为784*10
	    # X的元素个数已知，-1自动匹配行数，softmax获得和为1的概率分布
	    return softmax(nd.dot(X.reshape((-1,num_inputs)),W) + b)

**3.6.5 定义损失函数**

	# y_hat为两个样本在3个类别的预测概率
	y_hat = nd.array([[0.1,0.3,0.6],[0.3,0.2,0.5]])
	# y为这两个样本的标签类别
	# NDArray默认为float类型，dtype设置为int32
	y =nd.array([0,2],dtype='int32')
	# 取得y_hat中的y对应下标的元素，默认按行取
	nd.pick(y_hat,y)

	out:
		[0.1 0.5]
		<NDArray 2 @cpu(0)>
	# 交叉熵损失函数
	def cross_entropy(y_hat,y):
	    return -nd.pick(y_hat,y).log()

**3.6.6 计算分类准确率**

分类准确率即正确预测数量与总预测数量之比。 

	def accuracy(y_hat,y):
	    # y_hat为预测概率，y为标签
	    # y_hat.argmax(axis=1)返回每行最大元素的索引，axis=0返回每列最大元素的索引，一行为一个样本
	    # astype设置为float32类型，y_hat为float类型      # 若==则代表模型预测准确
	    return (y_hat.argmax(axis=1) == y.astype('float32')).sum().asscalar()/y.size
	    #return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar() #等价

我们可以评价模型net在数据集data_iter上的准确率

	# 本函数已保存在d2l包中方便以后使用
	def evaluate_accuracy(data_iter,net):
	    acc_sum, n=0.0, 0
	    for X,y in data_iter:
	        y = y.astype('float32') # 标签y设置为float型
	        # net(X)为模型预测概率值
	        acc_sum +=(net(X).argmax(axis=1) == y).sum().asscalar()
	        n +=y.size
	    return acc_sum/n

**3.6.7 训练模型**

	num_epochs,lr =5, 0.1
	
	# 本函数已保存在d2lzh包中方便以后使用
	def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,trainer=None):
	    for epoch in range(1,num_epochs+1):
	        train_l_sum,train_acc_sum,n=0.0,0.0,0
	        # 对小批量样品进行训练   行数为小批量样本数
	        for X,y in train_iter:
	            with autograd.record():
	                y_hat = net(X)
	                l =loss(y_hat,y).sum() #小批量样品的损失函数和
	            l.backward()
	            if trainer is None: # 默认优化算法为sgd
	                d2l.sgd(params,lr,batch_size)
	            else:
	                train.step(batch_size)
	            y = y.astype('float32')
	            train_l_sum +=l.asscalar()# 所有样品的损失函数的和
	            train_acc_sum +=(y_hat.argmax(axis=1)  == y).sum().asscalar()# 所有样品预测准确的个数
	            n += y.size
	        test_acc =evaluate_accuracy(test_iter,net)
	        print('epoch %d,loss %.4f,train acc %.3f,test acc %.3f' %(epoch,train_l_sum/n,train_acc_sum/n,test_acc))
    
**练习**

本节中的cross_entropy函数是按照“softmax回归”一节中的交叉熵损失函数的数学定义实现的。这样的实现方式可能有什么问题？（提示：思考一下对数函数的定义域。） 

	对数函数要求模型输出值y_hat必须是非负数，且y_hat中的每个元素必须是在0-1之间，否则交叉熵损失函数的值是一个负数

**3.7 softmax回归的简洁实现 **

	# 1. 首先导入所需的包或模块
	%matplotlib inline
	import d2lzh as d2l
	from mxnet import gluon,init
	from mxnet.gluon import loss as gloss,nn
	
	# 2. 获取和读取数据
	batch_size =256
	train_iter,test_iter =d2l.load_data_fashion_mnist(batch_size)
	
	# 3. 定义和初始化模型
	# softmax回归的输出层是⼀个全连接层。因此，我们添加⼀个输出个数为10的全连接层。
	net = nn.Sequential()
	net.add(nn.Dense(10))
	net.initialize(init.Normal(sigma=0.01)) # 初始化模型参数 
	
	# 4. softmax和交叉熵损失函数
	# 分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定。
	loss =gloss.SoftmaxCrossEntropyLoss()
	
	# 5. 定义优化算法
	# 使用学习率为0.1的小批量随机梯度下降作为优化算法。
	trainer =gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.1})
	
	# 6. 训练模型
	num_epochs=5
	d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)











