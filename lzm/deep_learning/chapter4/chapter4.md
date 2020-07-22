#第四章 深度学习计算#
	
**4.1 模型构造**

**1. 继承Block类来构造模型**

Block类是nn模块里提供的一个模型构造类，是所有神经网络层和模型的基类。我们可以继承它来定义我们想要的模型。

	from mxnet import nd
	from mxnet.gluon import nn
	# 定义的MLP类重载了Block类的__init__函数和forward函数。
	# 它们分别用于创建模型参数和定义前向计算。前向计算也即正向传播。 
	class MLP(nn.Block):
	    # 声明带有模型参数的层，这里声明了两个全连接层 
	    def __init__(self,**kwargs):
	        # 调用MLP的父类Block的构造函数来进行必要的初始化(如_children)。这样在构造实例时还可以指定其他函数参数
	        super(MLP,self).__init__(**kwargs)
	        self.hidden=nn.Dense(256,activation='relu')# 隐藏层 
	        self.output=nn.Dense(10)# 输出层 
	    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出     
	    def forward(self,x):
	        return self.output(self.hidden(x))

以上的MLP类中无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的backward函数。     
	
	# uniform默认为[0,1)的均匀分布
	X=nd.random.uniform(shape=(2,20))
	net=MLP()
	net.initialize()
	# net(X)会调用MLP继承自Block类的__call__函数，这个函数将调用MLP类定义的forward函数来完成前向计算。 
	net(X)
	

**2. Sequential类继承自Block类**

	# 实现一个与Sequential类有相同功能的MySequential类。
	class MySequential(nn.Block):
	    # 调用MySequential的父类Block的构造函数来进行必要的初始化
	    def __init__(self,**kwargs):
	        super(MySequential,self).__init__(**kwargs)
	        
	    def add(self,block):
	        # block是一个Block子类实例，假设它有一个独一无二的名字。我们将它保存在Block类的 
	        # 成员变量_children里，其类型是OrderedDict(字典)。当MySequential实例调⽤ 
	        # initialize函数时，系统会自动对_children里所有成员初始化
	        self._children[block.name]=block
	        
	    def forward(self,x):
	        # OrderedDict保证会按照成员添加时的顺序遍历成员 
	        # values()返回OrderedDict的值(即层)
	        for block in self._children.values():
	            x=block(x)
	        return x
	
**3. 构造复杂的模型**

	class FancyMLP(nn.Block):
	    def __init__(self,**kwargs):
	        super(FancyMLP,self).__init__(**kwargs)
	        # 使用get_constant创建的随机权重参数不会在训练中被迭代（即常数参数）
	        # params为参数字典(ParameterDict)，get_constant检索参数字典，查找'rand_weight',
	        # 若未找到，则在参数字典中创建键值对(key='rand_weight',value为Constant类型)
	        self.rand_weight=self.params.get_constant('rand_weight',nd.random.uniform(shape=(20,20)))
	        self.dense=nn.Dense(20,activation='relu')
	        
	    def forward(self,x):
	        x=self.dense(x)
	        # 使⽤创建的常数参数，以及NDArray的relu函数和dot函数
	        x=nd.relu( nd.dot(x,self.rand_weight.data())+1 )
	        # 复用全连接层。等价于两个全连接层共享参数 
	        x=self.dense(x)
	        # 控制流，这⾥我们需要调用asscalar函数来返回标量进行比较
	        while x.norm().asscalar()>1:
	            x/=2
	        if x.norm().asscalar()<0.8:
	            x*=10
	        return x.sum()

因为FancyMLP和Sequential类都是Block类的子类，所以我们可以嵌套调⽤它们。 

	class NestMLP(nn.Block):
	    def __init__(self,**kwargs):
	        super(NestMLP,self).__init__(**kwargs)
	        self.net=nn.Sequential()
	        self.net.add(nn.Dense(64,activation='relu'),
	                    nn.Dense(32,activation='relu'))
	        self.dense=nn.Dense(16,activation='relu')
	        
	    def forward(self,x):
	        return self.dense(self.net(x))
	    
	net=nn.Sequential()
	net.add(NestMLP(),nn.Dense(20),FancyMLP())

**练习**

1.如果不在MLP类的`__init__`函数里调用父类的`__init__`函数，会出现什么样的错误信息？ 

	'MLP' object has no attribute '_children'
	_children未初始化

2.如果去掉FancyMLP类里面的asscalar函数，会有什么问题？ 
	
	无影响，广播机制会自动将标量转换成NDArray，以实现比较

3.如果将NestMLP类中通过Sequential实例定义的self.net改为`self.net = [nn. Dense(64, activation='relu'), nn.Dense(32, activation='relu')]`，会有什么问题？

	会报错，'list' object has no attribute 'add'

**4.2 模型参数的访问、初始化和共享** 
	
	from mxnet import init,nd
	from mxnet.gluon import nn
	
	net=nn.Sequential()
	net.add(nn.Dense(256,activation='relu'))
	net.add(nn.Dense(10))
	net.initialize()# 使⽤默认初始化方式,为[-0.07,0.07]的均匀分布(即init.Uniform(scale=0.7))
	
	X=nd.random.uniform(shape=(2,20))
	Y=net(X) # 前向计算 
	
**1. 访问模型参数** 

可以通过Block类的params属性来访问该层包含的所有参数，得到一个由参数名称映射到参数实例（Parametre类）的字典（类型为ParameterDict类）。

	net[0].params
	
可以通过名字（'dense0_weight'）来访问字典（类型为ParameterDict类）里的元素，也可以直接使用它的变量名（weight）。

	net[0].params['dense0_weight'],net[0].weight
	
Gluon里参数类型为Parameter类，它包含参数和梯度的数值，可以分别通过data函数和grad函数来访问。

	net[0].weight.data()
	# 权重梯度的形状和权重的形状⼀样。
	net[0].weight.grad()	
	net[1].bias.data()
	
使用collect_params函数来获取net变量所有嵌套（例如通过add函数嵌套）的层所包含的所有参数。它返回的同样是一个由参数名称到参数实例的字典（ParameterDict类）。 

	net.collect_params()

collect_params可以通过正则表达式来匹配参数名，从而筛选需要的参数。

	net.collect_params('.*weight')
	
**2. 初始化模型参数** 

net.initialize()为默认初始化方法，即（init.Uniform(scale=0.07)）权重参数元素为[0.07,0.07]之间均匀分布的随机数，偏差参数则全为0。

init.Normal(sigma=0.01)将权重参数初始化成均值为0、标准差为0.01的正态分布随机数，并依然将偏差参数清零。

	# 非首次对模型初始化需要指定force_reinit为真，代表强制重新初始化，默认False
	net.initialize(init=init.Normal(sigma=0.01),force_reinit=True)
	
init.Constant使用常数来初始化权重参数。

	net.initialize(init=init.Constant(1),force_reinit=True)

只对某个特定参数进行初始化。调用Parameter类的initialize函数，它 与Block类提供的initialize函数的使用方法一致致。

	# net[0].weight为Parameter类
	net[0].weight.initialize(init=init.Xavier(),force_reinit=True)
	
**3. 自定义初始化方法**

实现⼀个Initializer类的子类，重载`_init_weight`这个函数，即可实现自定义初始化。

	# init.Uniform、Normal、Xavier均为Initialize的子类
	class MyInit(init.Initializer):
	    def _init_weight(self,name,data):
	        print('Init',name,data.shape)
	        data[:]=nd.random.uniform(low=-10,high=10,shape=data.shape)
	        data*=data.abs()>=5
	        
	net.initialize(MyInit(),force_reinit=True)
	
	# 通过Parameter类的set_data函数来直接改写模型参数。
	net[0].weight.set_data(net[0].weight.data()+1)

**4. 共享模型参数** 

在构造第三隐藏层时通过params来指定它使用第二隐藏层的参数，因此模型的第二隐藏层（shared变量）和第三隐藏层共享模型参数。  
因为模型参数里包含了梯度，所以在反向传播计算时，第二隐藏层和第三隐藏层的梯度都会被累加在shared.params.grad()⾥。 

	net=nn.Sequential()
	shared=nn.Dense(8,activation='relu')
	net.add(nn.Dense(8,activation='relu'),
	       shared,
	       nn.Dense(8,activation='relu',params=shared.params),
	       nn.Dense(10))

**练习**

1.尝试在net.initialize()后、net(X)前访问模型参数，观察模型参数的形状。

	net.initialize()后、net(X)前访问模型参数发现模型参数的shape为(1,0)，可见权重w实际的初始化发生在第一个正向传播过程中(即net(X)，net.initialize()后并未初始化)
	
2.构造一个含共享参数层的多层感知机并训练。在训练过程中，观察每一层的模型参数和梯度。
 
	import d2lzh as d2l
	from mxnet import gluon,init,autograd
	from mxnet.gluon import loss as gloss,nn
	
	net=nn.Sequential()
	shared=nn.Dense(8,activation='relu')
	net.add(nn.Dense(8,activation='relu'),
	        shared,
	        nn.Dense(8,activation='relu',params=shared.params),
	        nn.Dense(10))
	net.initialize(init.Normal(sigma=0.01))
	
	batch_size,num_epochs=256,5
	loss=gloss.SoftmaxCrossEntropyLoss()
	train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
	trainer=gluon.Trainer(net.collect_params(),'sgd',{'learning_rate':0.5})
	for epoch in range(num_epochs):
	        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
	        for X, y in train_iter:
	            with autograd.record():
	                y_hat = net(X)
	                l = loss(y_hat, y).sum()
	            l.backward()
	            if trainer is None:
	                sgd(params, lr, batch_size)
	            else:
	                trainer.step(batch_size)
	            y = y.astype('float32')
	            train_l_sum += l.asscalar()
	            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
	            n += y.size
	        test_acc = d2l.evaluate_accuracy(test_iter, net)
	        print(net[1].weight.data()==net[2].weight.data())

**4.3 模型参数的延后初始化**

**1.延后初始化**

延后初始化（deferredinitialization）：系统将真正的参数初始化延后到获得足够信息时才执行的行为。  
它可以让模型的创建更加简单：只需要定义每个层的输出⼤小，而不用人工推测它们的输入个数。

调用initialize函数时，由于隐藏层输入个数依然未知，系统也无法得知该层权重参数的形状，并没有真正初始化参数。只有在当我们将确定形状的输入X传进网络做前向计算net(X)时，系统才推断出该层的权重参数的形状。系统在创建这些参数之后，并进行初始化，然后才进行前向计算。 因此，net(X)时才能真正开始初始化参数。 

这个初始化只会在第一次前向计算时被调用。之后我们再运行前向计算net(X)时则不会重新初始化。
	
**2. 避免延后初始化** 

如果系统在调用initialize函数时能够知道所有参数的形状，那么延后初始化就不会发生。

	# 1.对已初始化的模型重新初始化时。因为参数形状不会发生变化，所以系统能够立即进行重新初始化。 
	net.initialize(init=MyInit(),force_reinit=True)
	
	# 2.通过in_units来指定每个全连接层的输入个数，使初始化能够在initialize函数被调用时立即发生。
	net=nn.Sequential()
	net.add(nn.Dense(256,in_units=20,activation='relu'))
	net.add(nn.Dense(10,in_units=256))
	
	net.initialize(init=MyInit())

**练习**

如果在下一次前向计算net(X)前改变输入X的形状，包括批量大小和输入个数，会发生什么？

	会报错，形状不一致(Shape inconsistent)

**4.4 自定义层** 

可以通过Block类自定义神经网络中的层，从而可以被重复调用。 

**1. 不含模型参数的自定义层**

	from mxnet import gluon,nd
	from mxnet.gluon import nn
	
	class CenteredLayer(nn.Block):
	    def __init__(self,**kwargs):
	        super(CenteredLayer,self).__init__(**kwargs)
	        
	    def forward(self,x):
	        return x-x.mean()
	# 实例化这个层，然后做前向计算
	layer=CenteredLayer()
	layer(nd.array([1,2,3,4,5]))
	# 构造更复杂的模型
	net=nn.Sequential()
	net.add(nn.Dense(128),
	       CenteredLayer())

2**. 含模型参数的自定义层**
 
	params=gluon.ParameterDict()
	# get在params(Block自带的成员变量，为ParameterDict类型)这个参数字典中查找'param2'，
	# 若未找到则创建键值对(key='param2',value为parameter类型)
	params.get('param2',shape=(2,3))
	
	class MyDense(nn.Block):
	    # units为该层的输出个数，in_units为该层的输入个数
	    def __init__(self,units,in_units,**kwargs):
	        super(MyDense,self).__init__(**kwargs)
	        self.weight=self.params.get('weight',shape=(in_units,units))
	        self.bias=self.params.get('bias',shape=(units,))
	        
	    def forward(self,x):
	        linear=nd.dot(x,self.weight.data())+self.bias.data()
	        return nd.relu(linear)
	# 实例化MyDense类并访问它的模型参数
	dense=MyDense(units=3,in_units=5)
	dense.params
	# 直接使用自定义层做前向计算
	dense.initialize()
	dense(nd.random.uniform(shape=(2,5)))
	# 使用自定义层构造模型
	net=nn.Sequential()
	net.add(MyDense(8,in_units=64),
	       MyDense(1,in_units=8))

**4.5 读取和存储**
 
**1. 读写NDArray**

通过save函数和load函数可以很方便地读写NDArray。 

	from mxnet import nd
	from mxnet.gluon import nn
	
	x=nd.ones(3)
	nd.save('x',x)
	x2=nd.load('x')
	
	y=nd.zeros(4)
	nd.save('xy',[x,y])# 保存NDArray列表为文件xy
	x2,y2=nd.load('xy')# 读取xy文件里的NDArray
	x2,y2
	
	mydict={'x':x,'y':y}
	nd.save('mydict',mydict)# 保存str->NDArray的字典为文件mydict
	mydict2=nd.load('mydict')
	
**2. 读写Gluon模型的参数**

通过`load_parameters`函数和`save_parameters`函数可以很方便地读写Gluon模型的参数。 

	class MLP(nn.Block):
	    def __init__(self,**kwargs):
	        super(MLP,self).__init__(**kwargs)
	        self.hidden=nn.Dense(256,activation='relu')
	        self.output=nn.Dense(10)
	        
	    def forward(self,x):
	        return self.output(self.hidden(x))
	    
	net=MLP()
	net.initialize()
	X=nd.random.uniform(shape=(2,20))
	Y=net(X)
	
	filename='mlp.params'
	net.save_parameters(filename)# 保存模型参数为文件
	
	net2=MLP()
	net2.load_parameters(filename)# 读取文件里的模型参数
	
**4.6 GPU计算**

 MXNet可以指定用来存储和计算的设备，如使用内存的CPU或者使用显存的GPU。在默认情况下，MXNet会将数据创建在内存，然后利用CPU来计算。 

**1. 计算设备**

	import mxnet as mx
	from mxnet import nd
	from mxnet.gluon import nn
	
	# mx.cpu()表示所有的物理CPU和内存。
	# mx.gpu()只代表⼀块GPU和相应的显存。
	mx.cpu(1),mx.gpu(),mx.gpu(1)
	
**2. NDArray的GPU计算**

在默认情况下，NDArray存在内存上。通过NDArray的context属性来查看该NDArray所在的设备。 

	x=nd.array([1,2,3])
	x.context
	
**GPU上的存储**

1.在创建NDArray的时候通过ctx参数指定存储设备。

	a=nd.array([1,2,3],ctx=mx.gpu())
	B=nd.random.uniform(shape=(2,3),ctx=mx.gpu())

2.通过`copyto`函数和`as_in_context`函数在设备之间传输数据。
 
	# 将内存上的NDArray变量x复制到gpu(0)上。 x->gpu
	y=x.copyto(mx.gpu())
	z=x.as_in_context(mx.gpu())

copyto还可以复制NDArray变量m复制到NDArray变量n上(m->n),但m，n形状必须相同
	m=nd.array([1,1,1])
	n=nd.array([1,2,3])
	t=m.copyto(n)

如果源变量和目标变量的context一致，`as_in_context`函数使目标变量和源变量共享源变量的内存或显存。而`copyto`函数总是为目标变量开新的内存或显存。 

**GPU上的计算**

MXNet的计算会在数据的context属性所指定的设备上执行。为了使用GPU计算，我们只需要事先将数据存储在显存上。计算结果会自动保存在同⼀块显卡的显存上。 
	
MXNet要求计算的所有输入数据都在内存或同一块显卡的显存上。这样设计的原因是CPU和不同的GPU之间的数据交互通常比较耗时。  
当我们打印NDArray或将NDArray转换成NumPy格式时，如果数据不在内存里，MXNet会将它先复制到内存，从而造成额外的传输开销。 
	
**3. Gluon的GPU计算**

Gluon的模型可以在初始化时通过ctx参数指定设备。

	net=nn.Sequential()
	net.add(nn.Dense(1))
	net.initialize(ctx=mx.gpu())# 将模型参数初始化在显存上。 
	
当输入是显存上的NDArray时，Gluon会在同⼀块显卡的显存上计算结果。 
	


