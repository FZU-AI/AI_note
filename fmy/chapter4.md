### 自定义模型
```python
class MLP(nn.Block):
	'''
	自定义网络，均继承nn.Block，使用forward实现计算方式
	'''
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(32, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

```

### 类似keras的逐层添加
```python
class MySequential(nn.Block):
	def __init__(self, **kwargs):
		super(MySequential, self).__init__(**kwargs)
	def add(self, block):
		self._children[block.name] = block
	def forward(self, x):
		for block in self._children.values():
			x = block(x)
		return x
    
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)
```

### 模型保存与使用
```python
net = MLP() 
net.initialize() 
X = nd.random.uniform(shape=(2, 20)) 
Y = net(X)
# 存储参数
net.save_parameters('mlp.params') 
# 再次使用
net2 = MLP() 
net2.load_parameters('mlp.params') 
```