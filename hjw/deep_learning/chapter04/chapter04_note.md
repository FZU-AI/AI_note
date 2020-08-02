# 1.模型构建

**Block类**

- Block类是nn模块⾥提供的⼀个模型构造类 

- 类是⼀个可供⾃由组建的部件。它的⼦类既可以是⼀个层（如Gluon提供的Dense类），⼜可以是⼀个模型（如这⾥定义的MLP类），或者是模型的⼀个部分 

- 事实上， Sequential类继承⾃Block类 

---

**定义MLP类的过程：**

- 重载 init  函数，用于创建模型参数
- 重载forward函数，用于定义前向计算
- 无需定义反向函数，系统将通过⾃动求梯度而⾃动⽣成反向传播所需的backward函数。 

---

**可变参数**

> - 如果我们不确定要往函数中传入多少个参数，或者我们想往函数中以列表和元组的形式传参数时，那就使要用*args；
> - 如果我们不知道要往函数中传入多少个关键词参数，或者想传入字典的值作为关键词参数时，那就要使用**kwargs。

---

---

Sequential类的⽬的：它提供add函数来逐⼀添加串联的Block⼦类实例，而模型的前向计算就是将这些实例
按添加的顺序逐⼀计算。 

---

# 2.**模型的参数**

**访问模型参数**

> 通过[ ]来访问网络中的任意层，类似于数组或者列表，方括号[ ]内的数字为索引，从0开始。其中索引0表⽰隐藏层为Sequential实例最先添加的层 。

**参数格式**

> - 得到的是由参数名称映射到参数实例的字典（类型为ParameterDict类） 
> - 为了访问特定参数，我们既可以通过名字来访问字典⾥的元素，也可以直接使⽤它的变量名 ，两者是等价的，为了保证可读性通常直接使用变量名来进行访问。

**参数类型及值获取**

Gluon⾥参数类型为Parameter类，它包含参数和梯度的数值，可以分别通过data函数和grad函
数来访问 。

- data函数：参数的值
- grad函数：梯度的数值

collect_params函数来获取net变量所有嵌套（例如通过add函数嵌套）的层所包含的所有参数。它返回的同样是⼀个由参数名称到参数实例的字典。 并且支持正则表达式匹配。

---

**初始化参数模型**

- 使用默认的初始化方法
- 自定义初始化方法

**共享模型参数**

- 如果不同层使⽤同⼀份参数，那么它们在前向计算和反向传播时都会共享相同的参数 

- 模型参数⾥包含了梯度，所以在反向传播计算时，第⼆隐藏层和第三隐藏层的梯度都会被累加在shared.params.grad()⾥。 

---

# 3.参数延后初始化

**延后初始化:**

- 系统将真正的参数初始化延后到获得足够信息时才执行的行为叫作延后初始化（deferred initialization）

- 延后初始化可以使模型的创建更加简单，只要定义每个层的输出大小，不需要人工推测输入个数
- 在进行第一次前向计算时，无法对模型参数进行操作，所以会额外做一次前向计算来迫使参数完成初始化

---

**避免延后初始化：**

系统在调⽤initialize函数时能够知道所有参数的形状，那么延后初始化就不会发⽣。 

- 对已初始化的模型重新初始化时 
- 在创建层的时候指定了它的输⼊个数，使系统不需要额外的信息来推测参数形
  状。 

# 4.自定义层

- 不含模型参数的自定义层
- 含模型参数的自定义层

自定义层跟nn模块中的其他层一样，可用于构造模型，前向计算等。

---

# 5.读取与存储

主要包括对数据的读取与存储（相关文件操作）

---

save函数：将变量存入文件中

nd.save('x',x) : 将变量x存在⽂件名同为x的⽂件⾥ 

---

load函数：从文件中读取数据

nd.load('x') : 从文件名为‘x'的文件中读取数据

---

**可以读写数组，字典，字符串，以及模型的参数**

---

# 6.GPU计算

**问题踩坑**

配置mxnet gpu 环境出现问题，一直加载不了gpu

后来发现自己的垃圾显卡只支持cuda9.2，而下载的确实cuda10.1,版本不对导致的问题

---

**计算设备**

MXNet可以指定⽤来存储和计算的设备，如使⽤内存的CPU或者使⽤显存的GPU 。默认为cpu,⽤mx.gpu(i)来表⽰第i块GPU及相应的显存（i从0开始）且mx.gpu(0)和mx.gpu()等价。 

**GPU计算**

创建ndarray和模型计算时都可以通过ctx=mx.gpu() 指定使用GPU来进行计算