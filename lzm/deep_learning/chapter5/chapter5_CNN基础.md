# 第五章 卷积神经网络（CNN） #

**5.1 二维卷积层**

卷积神经网络（convolutional neural network）是含有卷积层（convolutional layer）的神经网络。  
二维卷积层有高和宽两个空间维度，常用来处理图像数据。

**1.二维互相关运算**

二维卷积层的核心计算是二维互相关运算。在二维卷积层中，一个二维输入数组和一个二维核（kernel）数组通过互相关运算输出一个二维数组。  
核数组在卷积计算中又称卷积核或过滤器（filter）。卷积核窗口（又称卷积窗口）的形状取决于卷积核的高和宽。  
在二维互相关运算中，卷积窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输入数组上滑动。当卷积窗口滑动到某一位置时，窗口中的输入子数组与核数组按元素相乘并求和，得到输出数组中相应位置的元素。

			 输入           核            输出
		
			0  1  2
		                   0  1         19  25
			3  4  5   *            =
		                   2  3         37  43
			6  7  8
	-----------------------------------------------------
			0×0 + 1×1 + 3×2 + 4×3 = 19, 
			1×0 + 2×1 + 4×2 + 5×3 = 25, 
			3×0 + 4×1 + 6×2 + 7×3 = 37, 
			4×0 + 5×1 + 7×2 + 8×3 = 43.

上述过程实现在corr2d函数里，它接受输入数组X与核数组K，并输出数组Y。 
	from mxnet import autograd,nd
	from mxnet.gluon import nn
	
	# 本函数已保存在d2l
	# corr2d使用了对单个元素赋值（[i, j]=）的操作因而⽆法⾃动求梯度。
	def corr2d(X,K):# 输入数组X(3x3)与核数组K(2x2)
	    h,w=K.shape
	    Y=nd.zeros((X.shape[0]-h+1,X.shape[1]-w+1))# 输出数组Y(2x2)
	    for i in range(Y.shape[0]):# 卷积窗口按行滚动
	        for j in range(Y.shape[1]):# 按列滚动
	            Y[i,j]=(X[i:i+h,j:j+w]*K).sum()
	    return Y

**2. 二维卷积层** 

二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。卷积层的模型参数包括了卷积核和标量偏差。  
在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差。

	class Conv2D(nn.Block):
	    def __init__(self,kernel_size,**kwargs):
	        super(Conv2D,self).__init__(**kwargs)
	        self.weight=self.params.get('weight',shape=kernel_size)# 卷积核
	        self.bias=self.params.get('bias',shape=(1,))# 偏差
	        
	    def forward(self,x):
	        return corr2d(x,self.weight.data())+self.bias.data()
	
**3. 图像中物体边缘检测** 

卷积层的简单应用：检测图像中物体的边缘，即找到像素变化的位置。
	
	# 当卷积核K与输入做互相关运算时，如果横向相邻元素相同，输出为0；否则输出为⾮0。 
	K=nd.array([[1,-1]])# K.shape为(1,2)
	K1=nd.array([1,-1])#  K1,shape为(2,)
	
	Y=corr2d(X,K)# 0到1和1到0的边缘检测为1、-1
	
**4. 通过数据学习核数组**

首先构造一个卷积层，将其卷积核初始化成随机数组。接下来在每一次迭代中，我们使用平方误差来比较Y和卷积层的输出Y_hat，然后计算梯度来更新权重（即卷积核）。

	# 构造一个输出通道数为1，核数组形状是(1, 2)的二维卷积层 
	conv2d=nn.Conv2D(1,kernel_size=(1,2))
	conv2d.initialize()
	
	# 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通道数均为1 
	X=X.reshape((1,1,6,8))
	Y=Y.reshape((1,1,6,7))
	
	for i in range(10):
	    with autograd.record():
	        Y_hat=conv2d(X)
	        l=(Y_hat-Y)**2
	    l.backward()
	    # 简单起见，这里忽略了偏差
	    conv2d.weight.data()[:] -=3e-2 *conv2d.weight.grad()
	    if(i+1)%2 == 0:
	        print('batch %d,loss %.3f' %(i+1,l.sum().asscalar()))
	
	conv2d.weight.data().reshape((1,2))#weight的shape为(1,1,1,2)

**5. 互相关运算和卷积运算** 

为了得到卷积运算的输出，我们只需将核数组左右翻转并上下翻转，再与输入数组做互相关运算。  
在深度学习中核数组都是学出来的：卷积层无论使用互相关运算或卷积运算都不影响模型预测时的输出。因此卷积层能使用互相关运算替代卷积运算。

**6. 特征图和感受野**

特征图 （feature map）：二维卷积层输出的二维数组可以看作是输入在空间维度（宽和高）上某一级的表征。  
感受野（receptive field）：影响特征图上的元素x的前向计算的所有可能输入区域（可能大于输入的实际尺寸）叫做x的感受野。  
通过更深的卷积神经网络使特征图中单个元素的感受野变得更加广阔，从而捕捉输入上更大尺寸的特征。 

**练习**

1.构造一个输入图像X，令它有水平方向的边缘。如何设计卷积核K来检测图像中水平边缘？ 如果是对角方向的边缘呢？ 

	# 水平方向的边缘
	X1=nd.ones((6,8))
	X1[2:4,:]=0
	K=nd.array([1,-1]).reshape(2,1)
	Y1=corr2d(X1,K)
	X1,Y1
	
	# 对角方向的边缘
	X1=nd.ones((6,8))
	for i in range(min(X1.shape)):
	    X1[i,i]=0
	K=nd.array([1,1,-1,-1]).reshape(2,2)
	Y1=corr2d(X1,K)
	X1,Y1

**5.2 填充和步幅** 

一般来说，假设输入形状是n<sub>h</sub>×n<sub>w</sub>，卷积核窗口形状是k<sub>h</sub>×k<sub>w</sub>，那么输出形状将会是   
(n<sub>h</sub> −k<sub>h</sub> + 1)×(n<sub>w</sub> −k<sub>w</sub> + 1)。`# n-k+1`   
所以卷积层的输出形状由输入形状和卷积核窗口形状决定。本节我们将介绍卷积层的两个超参数，即填充和步幅。它们可以对给定形状的输入和卷积核改变输出形状。 

**1. 填充**

填充（padding）：在输入高和宽的两侧填充元素（通常是0元素）。 填充可以增加输出的高和宽，这常用来使输出与输入具有相同的高和宽。 


			 输入           核            输出
		
         0  0  0  0  0
                                       0  3  8  4 
		 0  0  1  2  0
		                   0  1        9  19 25 10
 	     0	3  4  5  0  *            =
		                   2  3        21 37 43 16
		 0  6  7  8  0
                                       6  7  8  0
         0  0  0  0  0
	-----------------------------------------------------
	     0×0+0×1+0×2+0×3 = 0

一般来说，如果在高的两侧一共填充p<sub>h</sub>行，在宽的两侧一共填充p<sub>w</sub>列，那么输出形状将会是  
(n<sub>h</sub> −k<sub>h</sub> + p<sub>h</sub> + 1)×(n<sub>w</sub> −k<sub>w</sub> + p<sub>w</sub> + 1)，`# n-k+p+1`  
也就是说，输出的高和宽会分别增加p<sub>h</sub>和p<sub>w</sub>。  
在很多情况下，我们会设置p<sub>h</sub> = k<sub>h</sub>−1和p<sub>w</sub> = k<sub>w</sub>−1来使输入和输出具有相同的高和宽。  

如果k<sub>h</sub>是奇数，我们会在高的两侧分别填充p<sub>h</sub>/2行。   
如果k<sub>h</sub>是偶数，一种可能是在输入的顶端一侧填充⌈p<sub>h</sub>/2⌉行，而在底端一侧填充⌊p<sub>h</sub>/2⌋行。

**2. 步幅**

步幅（stride）：卷积窗口每次滑动的行数和列数。步幅可以减小输出的高和宽，例如输出的高和宽仅为输入的高和宽的1/n（n为大于1的整 数）。   

一般来说，当高上步幅为s<sub>h</sub>，宽上步幅为s<sub>w</sub>时，输出形状为  
⌊(n<sub>h</sub> −k<sub>h</sub> + p<sub>h</sub> + s<sub>h</sub>)/s<sub>h</sub>⌋×⌊(n<sub>w</sub> −k<sub>w</sub> + p<sub>w</sub> + s<sub>w</sub>)/s<sub>w</sub>⌋。`# (n-k+p+s)/s `  
如果设置p<sub>h</sub> = k<sub>h</sub>−1和p<sub>w</sub> = k<sub>w</sub>−1，那么输出形状将简化为  
⌊(n<sub>h</sub>+s<sub>h</sub>−1)/s<sub>h</sub>⌋×⌊(n<sub>w</sub>+s<sub>w</sub>−1)/s<sub>w</sub>⌋。

当输入的高和宽两侧的填充数分别为p<sub>h</sub>和p<sub>w</sub>时，我们称填充为(p<sub>h</sub>,p<sub>w</sub>)。特别地， 当p<sub>h</sub> = p<sub>w</sub> = p时，填充为p。  
当在高和宽上的步幅分别为s<sub>h</sub>和s<sub>w</sub>时，我们称步幅为(s<sub>h</sub>,s<sub>w</sub>)。特别 地，当s<sub>h</sub> = s<sub>w</sub> = s时，步幅为s。  
在默认情况下，填充为0，步幅为1。 

**练习**

1.对本节最后一个例子通过形状计算公式来计算输出形状，看看是否和实验结果一致。 

	n=(8,8),k=(3,5),p=(0,1),s=(3,4)
	⌊8-3+0+3/3⌋=2,⌊8-5+1+4/4⌋=2 # ⌊n-k+p+s/s⌋ 

**5.3 多输入通道和多输出通道**

彩色图像在高和宽2个维度外还有RGB（红、绿、蓝）3个颜色通道。假设彩色图像的高和宽分别是h和w（像素），那么它可以表示为一个3×h×w的多维数组。我们将大小为3的这一维称为**通道**（channel）维。 使用多通道可以拓展卷积层的模型参数。 

**1. 多输入通道**

假设输入数据的通道数为c<sub>i</sub>，那么卷积核的输入通道数同样为c<sub>i</sub>。由于输入和卷积核各有c<sub>i</sub>个通道，我们可以在各个通道上对输入的二维数组和卷积核的二维核数组做互相关运算，再将这c<sub>i</sub>个互相关运算的二维输出按通道相加，得到一个二维数组。
	
	import d2lzh as d2l
	from mxnet import nd
	# 实现含多个输入通道的互相关运算。我们只需要对每个通道做互相关运算，然后通过add_n函数来进行累加。 
	def corr2d_multi_in(X,K):
	    # 首先沿着X和K的第0维（通道维）遍历（for x,k in zip(X,K)得到每个输入通道上的输入数组x,核数组k）。
	    # d2l.corr2d(x,k)得到每个输入通道上的输出数组
	    # 然后使用*将输出数组列表变成add_n函数的位置参数 （positional argument）来进行相加
	    return nd.add_n(*[d2l.corr2d(x,k) for x,k in zip(X,K)])
	
**2. 多输出通道**

如果希望得到含多个通道的输出，我们可以为每个输出通道分别创建形状为c<sub>i</sub> ×k<sub>h</sub> ×k<sub>w</sub>的核数组。将它们在输出通道维上连结，卷积核的形状即c<sub>o</sub> ×c<sub>i</sub> ×k<sub>h</sub> ×k<sub>w</sub>。

	# 实现一个互相关运算函数来计算多个通道的输出。 
	def corr2d_multi_in_out(X,K):
	    # 每个输出通道上的结果由卷积核在该输出通道上的核数组(k)与整个输入数组(X)计算而来。 
	    # for k in K 得到每个输出通道上的核数组k，corr2d_multi_in(X,k)的得到每个输出通道的输出数组的合并在一起
	    # 通过nd.stack(*)将每个输出通道的输出数组的合并在一起
	    return nd.stack(*[corr2d_multi_in(X,k) for k in K])
	
	x=nd.array([1,2,3]).reshape(1,1,3)
	y=nd.array([4,5,6]).reshape(1,1,3)
	z=nd.array([7,8,9]).reshape(1,1,3)
	nd.stack(x,y,z,axis=0)#按第一维合并，合并后的shape=3（第一维）x1x1x3

**3. 1×1卷积层**

1×1卷积层：卷积窗口形状为1×1（k<sub>h</sub> = k<sub>w</sub> = 1）的多通道卷积层。其中的卷积运算称为1×1卷积。  
输入X的形状为c<sub>i</sub> × h × w ，卷积核K的形状为c<sub>o</sub> ×c<sub>i</sub> ×1 ×1。
假设我们将通道维当作特征维，将高和宽维度上的元素当成数据样本，那 么1×1卷积层的作用与全连接层等价。  
**1×1卷积层可以被当作保持高和宽维度形状不变的全连接层使用。**  
1×1卷积层通常用来调整网络层之间的通道数，并控制模型复杂度。 

	# 输出中的每个元素来自输入中在高和宽上相同位置的元素在不同通道之间的按权重累加。
	def corr2d_multi_in_out_1x1(X,K):
	    c_i,h,w=X.shape # 输入通道数，高，宽
	    c_o=K.shape[0] # 输出通道数
	    X=X.reshape((c_i,h*w)) #X的形状为c_i × h × w 
	    K=K.reshape((c_o,c_i)) #K的形状为c_o x c_i x 1 x 1
	    Y=nd.dot(K,X) # 全连接层的矩阵乘法 
	    return Y.reshape((c_o,h,w))

**5.4 池化层**

池化（pooling）层的一个主要作用是：可以缓解卷积层对位置的过度敏感性。

**1. 二维最大池化层和平均池化层** 

同卷积层一样，池化层每次对输入数据的一个固定形状窗口（又称池化窗口）中的元素计算输出。  
最大池化或平均池化：池化层直接计算池化窗口内输入元素的最大值或者平均值。  
在二维最大池化中，池化窗口从输入数组的最左上方开始，按从左往右、从上往下的顺序，依次在输入数组上滑动。当池化窗口滑动到某一位置时，池化窗口中的输入子数组的最大值即输出数组中相应位置的元素。  
二维平均池化的工作原理与二维最大池化类似，但将最大运算符替换成平均运算符。  
池化窗口形状为p×q的池化层称为p×q池化层，其中的池化运算叫作p×q池化。 

	最大池化

			  输入                       输出
		
			0  1  2
    	                   2 x 2          4  5
			3  4  5         最大    
		                   池化层           7  8
			6  7  8
	-----------------------------------------------------
					max(0,1,3,4) = 4, 
					max(1,2,4,5) = 5,
					max(3,4,6,7) = 7,
					max(4,5,7,8) = 8.

使用2×2最大池化层时，只要卷积层识别的模式在高和宽上移动不超过一个元素，我们依然可以将它检测出来。

	# 池化层的前向计算实现在pool2d函数里。
	from mxnet import nd
	from mxnet.gluon import nn
	
	def pool2d(X,pool_size,mode='max'):
	    p_h,p_w=pool_size
	    Y=nd.zeros( (X.shape[0]-p_h+1,X.shape[1]-p_w+1) )# 输出数组2x2
	    for i in range(Y.shape[0]):# 池化窗口按行滚动
	        for j in range(Y.shape[1]):# 按列滚动
	            if mode=='max':
	                Y[i,j]=X[i:i+p_h,j:j+p_w].max()
	            elif mode=='avg':
	                Y[i,j]=X[i:i+p_h,j:j+p_w].mean()
	    return Y
	
**2. 填充和步幅**

同卷积层一样，池化层也可以在输入的高和宽两侧的填充并调整窗口的移动步幅来改变输出形状。池化层填充和步幅与卷积层填充和步幅的⼯作机制⼀样。

	X=nd.arange(16).reshape((1,1,4,4))# 前两个维度分别是批量和通道
	# 默认情况下，MaxPool2D实例里步幅和池化窗口形状相同(3x3)。
	pool2d=nn.MaxPool2D(3)
	pool2d(X) # 因为池化层没有模型参数，所以不需要调⽤参数初始化函数
	
	# 可以手动指定池化层的步幅和填充。 
	pool2d=nn.MaxPool2D(3,padding=1,strides=2)# (n-k+p+s)/s即(4-3+2+2)/2
	
	# 可以指定非正方形的池化窗口，并分别指定高和宽上的填充和步幅。 
	pool2d=nn.MaxPool2D((2,3),padding=(1,2),strides=(2,3))
	
**3. 多通道**

在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入数据卷积后再按通道相加。这意味着池化层的输出通道数与输入通道数相等。

	# nd.concat沿着连接轴的输出数组的尺寸将等于输入数组的相应维度(dim)的总和。
	X=nd.concat(X,X+1,dim=1)# 通道数变为2
	pool2d=nn.MaxPool2D(3,padding=1,strides=2)
	pool2d(X)# 池化后，输出通道数仍然是2。

**练习**

1.想⼀想，最大池化层和平均池化层在作用上可能有哪些区别？ 

	最大池化层获得图像的最大特征，而平均池化层获得图像的平均特征	

2.你觉得最小池化层这个想法有没有意义？

	没有意义，最小池化层获得图像的最小特征，而最小特征不能用于辨别区分物体

