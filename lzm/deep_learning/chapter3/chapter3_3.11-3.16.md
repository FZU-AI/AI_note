**3.11 模型选择、欠拟合和过拟合**

**1. 训练误差和泛化误差**

训练误差（trainingerror）：模型在训练数据集上表现出的误差。 

泛化误差（generalizationerror）：模型在任意一个测试数据样本上表 现出的误差的期望，并常常通过测试数据集上的误差来近似。 

训练误差可以认为是做往年高考试题（训练题）时的错误率，泛化误差则可以通过真正参加高考（测试题）时的答题错误率来近似。

训练误差的期望小于或等于泛化误差。即由训练数据集学到的模型参数会使模型在训练数据集上的表现优于或等于在测试数据集上的表现。由于无法从训练误差估计泛化误差，一味地降低训练误差并不意味着泛化误差一定会降低。   
**机器学习模型应关注降低泛化误差。** 

**2. 模型选择**

模型选择（model selection）:评估若干候选模型的表现并从中选择模型的过程。

验证数据集：预留一部分在训练数据集和测试数据集以外的数据来进行模型选择的数据集，简称验证集（validation set）。
我们可以从给定的训练集中随机选取一小部分作为验证集，而将剩余部分作为真正的训练集。 

K折交叉验证（K-foldcross-validation）：我们把原始训练数据集分割成K个不重合的⼦数据集，然后我们做K次模型训练和验证。每一次，我们使用一个子数据集验证模型，并使用其他K−1个子数据集来训练模型。最后，我们对这K次训练误差和验证误差分别求平均。 

**3. 欠拟合和过拟合** 

欠拟合（underfitting）：模型无法得到较低的训练误差

过拟合（overfitting）：模型的训练误差远小于它在测试数据集上的误差

影响欠拟合和过拟合的因素：模型复杂度和训练数据集大小。 

(1)模型复杂度：如果模型的复杂度过低，很容易出现欠拟合；如果模型复杂度过高，很容易出现过拟合。 应对欠拟合和过拟合的一个办法是针对数据集选择合适复杂度的模型。

高阶多项式函数比低阶多项式函数更容易在相同的训练数据集上得到更低的训练误差，即更易过拟合。

(2)训练数据集大小：如果训练数据集中样本 数过少，特别是比模型参数数量（按元素计）更少时，更易发生过拟合。此外，泛化误差不会随训练数据集里样本数量增加而增大。

**4. 多项式函数拟合实验** 

1.首先导入实验需要的包或模块

	%matplotlib inline
	import d2lzh as d2l
	from mxnet import autograd,gluon,nd
	from mxnet.gluon import data as gdata,loss as gloss,nn

2.生成数据集 

给定样本特征x，我们使用如下的三阶多项式函数来⽣成该样本的标签：  
y = 1.2x−3.4x<sup>2</sup> + 5.6x<sup>3</sup> + 5 + ϵ,  
其中噪声项ϵ服从均值为0、标准差为0.1的正态分布。训练数据集和测试数据集的样本数都设为100。 

    n_train,n_test,true_w,true_b=100,100,[1.2,-3.4,5.6],5
    features=nd.random.normal(shape=(n_train+n_test,1))
    # concat默认dim=1，按列连结两个矩阵
    ploy_features=nd.concat(features,nd.power(features,2),nd.power(features,3))
    labels =(true_w[0]*ploy_features[:,0]+true_w[1]*ploy_features[:,1])+true_w[2]*ploy_features[:,2]+true_b
    labels +=nd.random.normal(scale=0.1,shape=labels.shape)

3.定义、训练和测试模型 

	# 定义作图函数，以x和logy画图
	# 本函数已保存在d2lzh包中⽅便以后使⽤
	def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,legend=None,figsize=(3.5,2.5)):
	    d2l.set_figsize(figsize)
	    d2l.plt.xlabel(x_label)
	    d2l.plt.ylabel(y_label)
	    d2l.plt.semilogy(x_vals,y_vals)# 第一条曲线以线(默认)的形式画
	    if x2_vals and y2_vals:
	        d2l.plt.semilogy(x2_vals,y2_vals,linestyle=':')# 第二条曲线以点的形式画
	        d2l.plt.legend(legend)# 放置图例

尝试使用不同复杂度的模型来拟合生成的数据集。

	num_epochs,loss=100,gloss.L2Loss()
	
	def fit_and_plot(train_features,test_features,train_labels,test_labels):
	    net=nn.Sequential()
	    net.add(nn.Dense(1))
	    # 权重w实际的初始化发生在第一个正向传播过程中
	    net.initialize()# 默认初始化权重参数w为[-0.7,0.7]的范围,偏差b默认
	    batch_size=min(10,train_labels.shape[0])
	    # ArrayDataset将features和labels组合在一起，形成dataset。
	    # DataLoader从dataset中随机(shuffle)读取batch_size大小的小批量样本
	    train_iter=gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)
	    # Trainer用'sgd'算法优化参数集合,net.collect_params()收集所有参数
	    trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.01})
	    train_ls,test_ls=[],[]
	    for _ in range(num_epochs):# _ 忽略参数
	        for X,y in train_iter:
	            with autograd.record():
	                l=loss(net(X),y) # 权重w实际的初始化发生在第一个正向传播过程中
	            l.backward()
	            trainer.step(batch_size)
	        train_ls.append( loss(net(train_features),train_labels).mean().asscalar() )
	        test_ls.append( loss(net(test_features),test_labels).mean().asscalar() )
	    print('final epoch：train loss',train_ls[-1],'test loss',test_ls[-1])
	    semilogy(range(1,num_epochs+1),train_ls,'epoch','loss',
	             range(1,num_epochs+1),test_ls,['train','test'])
	    print('weight：',net[0].weight.data().asnumpy(),
	         '\nbias：',net[0].bias.data().asnumpy())

（1）三阶多项式函数拟合（正常）   
使用与数据生成函数同阶的三阶多项式函数拟合，实验表明，这个模型的训练误差和在测试数据集的误差都较低。

（2）线性函数拟合（欠拟合）  
线性模型在非线性模型（如三阶多项式函数）生成的数据集上容易欠拟合。 

（3）训练样本不足（过拟合）  
训练样本不足，这使模型显得过于复杂，以至于容易被训练数据中的噪声影响，容易发生过拟合。

应选择复杂度合适的模型并避免使用过少的训练样本。 

避免过拟合的常用方法：权重衰减和丢弃法(dropout)。

**3.12 权重衰减** 

**1. 方法**

权重衰减（weight decay）即L<sub>2</sub>范数正则化（regularization）化，通常会使学到的权重参数的元素较接近0。  
正则化通过为模型损失函数添加惩罚项使学出的模型参数值较小，是应对过拟合的常用手段 。

L<sub>2</sub>范数正则化在模型原损失函数基础上添加L2范数惩罚项，从而得到训练所需要最小化的损失函数。  
L<sub>2</sub>范数正则化令权重w先自乘小于1的数，再减去不含惩罚项的梯度。  
L<sub>2</sub>范数惩罚项指的是模型权重参数每个元素的平方和与一个正的常数的乘积。     
带有L<sub>2</sub>范数惩罚项的新损失函数为
ℓ(θ) =λ/2n *∑w<sub>i</sub><sup>2</sup>, 其中超参数λ > 0。  
当权重参数均为0时，惩罚项最小。当λ较大时，惩罚项在损失函数中的比重较大，这通常会使学到的权重参数的元素较接近0。当λ设为0时，惩罚项完全不起作用。  

**2. 简洁实现** 

	def fit_and_plot_gluon(wd):
	    net=nn.Sequential()
	    net.add(nn.Dense(1))
	    net.initialize(init.Normal(sigma=1))
	    # 通过wd参数来指定权重衰减超参数。默认下，Gluon会对权重和偏差同时衰减。
	    # 分别对权重和偏差构造Trainer实例，从而只对权重衰减。 
	    # 对权重参数衰减。net.collect_params(.*weight)收集所有以weight结尾的参数
	    trainer_w=gluon.Trainer(net.collect_params('.*weight'),'sgd',{'learning_rate':lr,'wd':wd})
	    # 不对偏差参数衰减。net.collect_params(.*bias)收集所有以weight结尾的参数
	    trainer_b=gluon.Trainer(net.collect_params('.*bias'),'sgd',{'learning_rate':lr})
	    train_ls,test_ls=[],[]
	    for _ in range(num_epochs):
	        for X,y in train_iter:
	            with autograd.record():
	                l=loss(net(X),y)
	            l.backward()
	            # 对两个Trainer实例分别调⽤step函数，从⽽分别更新权重和偏差 
	            trainer_w.step(batch_size)
	            trainer_b.step(batch_size)
	        train_ls.append( loss( net(train_features),train_labels ).mean().asscalar() )
	        test_ls.append( loss(net(test_features),test_labels ).mean().asscalar() )
	    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',
	                 range(1,num_epochs+1),test_ls,['train','test'])
	    print('L2 norm of w;',net[0].weight.data().norm().asscalar(),'L2 norm of b;',net[0].bias.data().norm().asscalar())

**练习**

1.回顾一下训练误差和泛化误差的关系。除了权重衰减、增大训练量以及使用复杂度合适的模型，你还能想到哪些办法来应对过拟合？ 

	dropout：以一定概率丢弃隐藏单元。
	由于在训练中隐藏层神经元的丢弃是随机的，即任意一个隐藏单元都有可能被清零，
	输出层的计算无法过度依赖任意一个隐藏单元，从而在训练模型时起到正则化的作用，用来应对过拟合。

2.调节实验中的权重衰减超参数，观察并分析实验结果。

	随着权重衰减超参数的增加，训练误差虽然有所提高，但测试集上的误差有所下降。
	过拟合现象得到一定程度的缓解。

**3.13 丢弃法（dropout）**

深度学习模型常常使用丢弃法（dropout）来应对过拟合问题，丢弃法只在训练模型时使⽤。  

**1. 方法**

单隐藏层的MLP，隐藏单元h<sub>i</sub>的计算表达式为  
h<sub>i</sub> = ϕ(x<sub>1</sub>w<sub>1i</sub> + x<sub>2</sub>w<sub>2i</sub> + x<sub>3</sub>w<sub>3i</sub> + x<sub>4</sub>w<sub>4i</sub> + b<sub>i</sub>),

设随机变量ξ<sub>i</sub>为0和1的概率分别为p和1−p。使用丢弃法时我们计算新的隐藏单元h<sub>i</sub>′  
 h<sub>i</sub>′ = ξ<sub>i</sub>h<sub>i</sub>/ 1−p，p为丢弃概率。

当ξ<sub>i</sub>=0时，h<sub>i</sub>′=0，代表隐藏单元i被抛弃。  
当ξ<sub>i</sub>=1时，h<sub>i</sub>′=h<sub>i</sub>/ 1−p，代表隐藏单元i被除以1−p做拉伸。

丢弃法不改变其输入的期望值。在反向传播时，与被丢弃的隐藏单元相关的权重的梯度均为0。由于在训练中隐藏层神经元的丢弃是随机的，即任意一个隐藏单元都有可能被清零，输出层的计算无法过度依赖任意一个隐藏单元，从而在训练模型时起到正则化的作用，并可以用来应对过拟合。在测试模型时，我们为了拿到更加确定性的结果，⼀般不使用丢弃法。

**2. 简洁实现** 

在Gluon中，我们只需要在全连接层后添加Dropout层并指定丢弃概率。在训练模型时， Dropout层将以指定的丢弃概率随机丢弃上⼀层的输出元素；在测试模型时，Dropout层并不发挥作用。 

	net=nn.Sequential()
	net.add(nn.Dense(256,activation="relu"),
	        nn.Dropout(drop_prob1),# 在第⼀个全连接层后添加丢弃层 
	        nn.Dense(256,activation="relu"),
	        nn.Dropout(drop_prob2),
	        nn.Dense(10) )
	net.initialize(init.Normal(sigma=0.01))
	# 下面训练并测试模型 
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':lr})
	d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,trainer)

**练习**

1.如果把本节中的两个丢弃概率超参数对调，会有什么结果？ 

	无明显变化

2.增大迭代周期数，比较使用丢弃法与不使用丢弃法的结果。
	
	对比发现，有丢弃法比无丢弃法的train acc更小，增长更缓慢，且均小于test acc，
	说明使用丢弃法可以避免过拟合
	
	有丢弃法
	epoch 1, loss 1.2322, train acc 0.521, test acc 0.773
	epoch 2, loss 0.5881, train acc 0.777, test acc 0.827
	epoch 3, loss 0.4993, train acc 0.816, test acc 0.846
	epoch 4, loss 0.4533, train acc 0.834, test acc 0.859
	epoch 5, loss 0.4244, train acc 0.846, test acc 0.858
	epoch 6, loss 0.4014, train acc 0.854, test acc 0.870
	epoch 7, loss 0.3871, train acc 0.859, test acc 0.873
	epoch 8, loss 0.3776, train acc 0.863, test acc 0.871
	epoch 9, loss 0.3665, train acc 0.866, test acc 0.874
	epoch 10, loss 0.3578, train acc 0.869, test acc 0.880

	无丢弃法
	epoch 1, loss 1.1086, train acc 0.568, test acc 0.794
	epoch 2, loss 0.5515, train acc 0.792, test acc 0.835
	epoch 3, loss 0.4541, train acc 0.831, test acc 0.853
	epoch 4, loss 0.4140, train acc 0.846, test acc 0.861
	epoch 5, loss 0.3829, train acc 0.858, test acc 0.871
	epoch 6, loss 0.3688, train acc 0.861, test acc 0.869
	epoch 7, loss 0.3483, train acc 0.871, test acc 0.869
	epoch 8, loss 0.3285, train acc 0.879, test acc 0.870
	epoch 9, loss 0.3194, train acc 0.881, test acc 0.879
	epoch 10, loss 0.3046, train acc 0.886, test acc 0.881
 
3.如果将模型改得更加复杂，如增加隐藏层单元，使⽤丢弃法应对过拟合的效果是否更加明显？ 

	隐藏单元数为256
	有丢弃法
	epoch 1, loss 1.1368, train acc 0.560, test acc 0.775
	epoch 2, loss 0.5751, train acc 0.784, test acc 0.838
	epoch 3, loss 0.4867, train acc 0.820, test acc 0.849
	epoch 4, loss 0.4479, train acc 0.835, test acc 0.857
	epoch 5, loss 0.4179, train acc 0.846, test acc 0.859
	无丢弃法
	epoch 1, loss 1.2566, train acc 0.512, test acc 0.716
	epoch 2, loss 0.5814, train acc 0.782, test acc 0.821
	epoch 3, loss 0.5081, train acc 0.815, test acc 0.839
	epoch 4, loss 0.4335, train acc 0.841, test acc 0.845
	epoch 5, loss 0.3958, train acc 0.855, test acc 0.859
	
	隐藏单元数为512
	有丢弃法
	epoch 1, loss 1.0415, train acc 0.599, test acc 0.790
	epoch 2, loss 0.5638, train acc 0.789, test acc 0.841
	epoch 3, loss 0.4730, train acc 0.826, test acc 0.857
	epoch 4, loss 0.4368, train acc 0.839, test acc 0.854
	epoch 5, loss 0.3993, train acc 0.854, test acc 0.865
	无丢弃法
	epoch 1, loss 1.1445, train acc 0.559, test acc 0.792
	epoch 2, loss 0.5450, train acc 0.797, test acc 0.832
	epoch 3, loss 0.4526, train acc 0.833, test acc 0.840
	epoch 4, loss 0.4132, train acc 0.846, test acc 0.855
	epoch 5, loss 0.4950, train acc 0.823, test acc 0.856

4.以本节中的模型为例，比较使用丢弃法与权重衰减的效果。如果同时使⽤丢弃法和权重衰减，效果会如何？

	以本节中的模型为例，实验发现，使用权重衰减效果并不好

**3.14 正向传播、反向传播和计算图** 

正向传播沿着从输入层到输出层的顺序，依次计算并存储神经网络的中间变量（包括输出）。

反向传播沿着从输出层到输入层的顺序，依次计算并存储神经网络的中间变量和参数的梯度。

正向传播和反向传播的相互依赖关系：  
正向传播的计算可能依赖于模型参数的当前值，而这些模型参数是在反向传播的梯度计算后通过优化算法迭代的。  
反向传播的梯度计算可能依赖于各变量的当前值，而这些变量的当前值是通过正向传播计算得到的。

**3.15 数值稳定性和模型初始化**

深度模型有关数值稳定性的典型问题是衰减（vanishing）和爆炸（explosion）。 当神经网络的层数较多时，模型的数值稳定性容易变差。 

**1. 衰减和爆炸**

假设一个层数为L的多层感知机的第l层**H**<sup>(l)</sup>的权重参数为**W**<sup>(l)</sup>。  
给定输入 **X**，多层感知机的第l层的输出**H** <sup>(l)</sup> =   **XW**<sup>(1)</sup>**W**<sup>(2)</sup> ...**W**<sup>(l)</sup>。此时，如果层数l较大，H(l)的计算可能会出现衰减或爆炸。  
举个例子，假设输⼊和所有层的权重参数都是标量，如权重参数为0.2和5，多层感知机的第30层输出为输入 **X**分别与0.2<sup>30</sup> ≈ 1×10<sup>−21</sup>（衰减）和5<sup>30</sup> ≈ 9×10<sup>20</sup>（爆炸）的乘积。类 似地，当层数较多时，梯度的计算也更容易出现衰减或爆炸。 

**2. 随机初始化模型参数** 

如果不对模型参数（特别是权重参数）进行随机初始化，在这种情况下，⽆无论隐藏单元有多少，隐藏层本质上只有1个隐藏单元在发挥作用。

(1)MXNet的默认随机初始化 

net.initialize(init.Normal(sigma=0.01))使模型net的权重参数采⽤正态分布的随机初始化方式。如果不指定初始化方法，如net.initialize()，MXNet将使⽤默认的随机初始化方法：权重参数每个元素随机采样于-0.07到0.07之间的均匀分布，偏差参数全部清零。 

(2)Xavier随机初始化 

Xavier随机初始化将使该层中权重参数的每个元素都随机采样于均匀分布。它的设计主要考虑到，模型参数初始化后，每层输出的方差不该受该层输入个数影响，且每层梯度的方差也不该受该层输出个数影响。

**3.16 实战Kaggle比赛：房价预测** 

**1. 获取和读取数据集** 

	%matplotlib inline
	import d2lzh as d2l
	from mxnet import autograd,gluon,init,nd
	from mxnet.gluon import data as gdata,loss as gloss,nn
	import numpy as np
	import pandas as pd
	
	#pd.read_csv读取文件，/相当于文件路径的\
	train_data=pd.read_csv('D:/data/train.csv')
	test_data=pd.read_csv('D:/data/test.csv')
	
	# 训练数据集包括1460个样本、80个特征和1个标签。 
	# 测试数据集包括1459个样本和80个特征。我们需要将测试数据集中每个样本的标签预测出来。 
	
	# 将所有的训练数据和测试数据的79个特征按样本连结。 pd.concat(,axis=0)axis=0(默认)为按行连结,axis=1为按列连结
	# iloc根据索引返回元素
	all_features=pd.concat( (train_data.iloc[:,1:-1],test_data.iloc[:,1:] ),axis=0)
	
**2. 预处理数据** 

	# all_features.dtypes返回每一列数据的类型
	# all_features.dtypes[all_features.dtypes!='object']为所有数据中类型不为object的数据(即均为数值型数据)，index返回索引
	numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
	# 标准化：将该特征的每个值先减去均值µ再除以标准差σ得到标准化后的每个特征值。
	all_features[numeric_features]=all_features[numeric_features].apply( lambda x:(x-x.mean())/(x.std()) )
	# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值(NaN),fillna(0)将缺失值换为0
	all_features[numeric_features]=all_features[numeric_features].fillna(0)
	
	# get_dummies将离散数值(如征MSZoning中RL、RM)转成指⽰特征(MSZoning_RL=1、MSZoning_RM=1)
	# dummy_na=True将缺失值也当作合法的特征值并为其创建指⽰特征 
	all_features=pd.get_dummies(all_features,dummy_na=True)
	
	n_train=train_data.shape[0]
	# values属性得到所有特征的值，且为NumPy格式的数据
	train_features=nd.array(all_features[:n_train].values)
	test_features=nd.array(all_features[n_train:].values)
	train_labels=nd.array(train_data.SalePrice.values).reshape((-1,1))
	
**3. 训练模型**

	loss=gloss.L2Loss()
	def get_net():
	    net=nn.Sequential()
	    net.add(nn.Dense(1))
	    net.initialize()
	    return net
	
	def log_rmse(net,features,labels):
	    # 将小于1的值设成1，使得取对数时数值更稳定 
	    # nd.clip(net(features),1,float('inf'))将net(features)中不在区间[1,inf]的数<1的设为1,>inf的设为inf
	    clipped_preds=nd.clip(net(features),1,float('inf'))
	    # loss里面有个1/2
	    rmse=nd.sqrt( 2*loss(clipped_preds.log(),labels.log() ).mean() )
	    return rmse.asscalar()
	
	def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
	    train_ls,test_ls=[],[]
	    train_iter=gdata.DataLoader(gdata.ArrayDataset(train_features,train_labels),batch_size,shuffle=True)
	    # 这里使用了Adam优化算法
	    trainer=gluon.Trainer(net.collect_params(),'adam',{'learning_rate':learning_rate,'wd':weight_decay})
	    for epoch in range(num_epochs):
	        for X,y in train_iter:
	            with autograd.record():
	                l=loss(net(X),y)
	            l.backward()
	            trainer.step(batch_size)
	        train_ls.append(log_rmse(net,train_features,train_labels))
	        if test_labels is not None:
	            test_ls.append(log_rmse(net,test_features,test_labels))
	    return train_ls,test_ls
	
**4. K折交叉验证**

K折交叉验证将被用来选择模型设计并调节超参数。


`get_k_fold_data()`返回第i折交叉验证时所需要的训练和验证数据。 

	def get_k_fold_data(k,i,X,y):
	    # 将原始数据集分为k个子集
	    assert k>1
	    fold_size=X.shape[0]//k
	    X_train,y_train=None,None
	    for j in range(k):
	        # slice创建一个切片对象
	        idx = slice(j*fold_size,(j+1)*fold_size)
	        X_part,y_part=X[idx,:],y[idx]
	        # 返回第i折交叉验证时所需要的训练和验证数据。
	        # 第i个为验证数据，其余的k-1个为训练数据 
	        if j==i:
	            X_valid,y_valid=X_part,y_part
	        elif X_train is None:
	            X_train,y_train=X_part,y_part
	        else:
	            X_train=nd.concat(X_train,X_part,dim=0)# dim=0按行连结
	            y_train=nd.concat(y_train,y_part,dim=0)
	    return X_train,y_train,X_valid,y_valid
	
在K折交叉验证中我们训练K次并返回训练和验证的平均误差。

	def k_fold(k,X_train,y_train,num_epochs,learning_rate,weight_decay,batch_size):
	    train_l_sum,valid_l_sum=0,0
	    for i in range(k):#进行k次训练和验证
	        data = get_k_fold_data(k,i,X_train,y_train)
	        net=get_net()
	        train_ls,valid_ls=train(net,*data,num_epochs,learning_rate,weight_decay,batch_size)
	        train_l_sum+=train_ls[-1]
	        valid_l_sum+=valid_ls[-1]
	        if i==0:
	            d2l.semilogy(range(1,num_epochs+1),train_ls,'epoch','rmse',
	                        range(1,num_epochs+1),valid_ls,['train','valid'])
	        print('fold %d,train rmse %f,valid rmse %f'%(i,train_ls[-1],valid_ls[-1]))
	    return train_l_sum/k,valid_l_sum/k
	
**5. 模型选择**

	k,num_epochs,lr,weight_decay,batch_size=5,100,5,0,64
	train_l,valid_l=k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
	print('%d-fold validation:avg train rmse %f,avg valid rmse %f'%(k,train_l,valid_l))
	
**6. 预测并在Kaggle提交结果**

	def train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size):
	    net=get_net()
	    train_ls,_=train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
	    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse')
	    print('train rmse %f' %train_ls[-1])
	    preds=net(test_features).asnumpy()
	    # 将值赋给'SalePrice'列，若test_data没有则增加这一列
	    test_data['SalePrice']=pd.Series(preds.reshape(1,-1)[0])
	    # pd.concat(,axis=1)axis=1表示按列连结'Id'、'SalePrice'两列
	    submission=pd.concat([ test_data['Id'],test_data['SalePrice'] ],axis=1)
	    # submission.to_csv会⽣成⼀个submission.csv⽂件
	    # index=False不写行名(索引),默认为True
	    submission.to_csv('submiss.csv',index=False)





​