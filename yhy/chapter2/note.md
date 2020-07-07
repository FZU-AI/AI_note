## 自动求梯度

- 梯度的计算
  $$
  假设有四个参数 \\
  X = [x_1,x_2,x_3,x_4] \\
  y = 2 * X.T * X \\
    = 2*(x_1^2 + x_2^2 + x_3^2 + x_4^2) \\
  y_{x_1}^{'} = 4(x_1) \\  
  y_{x_2}^{'} = 4(x_2) \\
  y_{x_3}^{'} = 4(x_3) \\
  y_{x_4}^{'} = 4(x_4) \\
  X的梯度为
  X = 4[x_1, x_2, x_3, x_4]
  $$
  

```python
from mxnet import autograd, nd
# x = [0,1,2,3]
x = nd.arange(4).reshape((4,1))

# x.attach_grad()申请存储梯度所需的内存
with autograd.record():# 记录梯度
    y = 2 * nd.dot(x.T, x)

# y.backward()自动求x的梯度
y.backward()

# 显示出x的梯度
x.grad
```



## 线性回归代码

```python
# 生成数据集
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:,0] + true_w[1] * features[:,1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 初始化模型
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()

# 定义模型
def linreg(X, w, b):
    return nd.dot(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size
        
# 训练模型
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X,y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        l.backward()
        sgd([w,b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch {},loss {}'.format(epoch+1, train_l.mean().asnumpy()))
```

