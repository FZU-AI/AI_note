# 第六章

与之前的神经网络不同，循环神经网络是为了更好的处理时序信息而设计的。它引入状态变量来存储过去的信息，并用其与当前的输入共同决定当前的输出。循环神经网络常用于处理序列数据，应用于语言模型、文本分类、机器翻译、语音识别等等。

### 6.1  语言模型

自然语言处理中最常见的是文本数据，语言模型是自然语言处理的重要技术。假设一段长度为T的文本词为w1，w2，w3，...，wT。当序列长度增加时，计算和存储的复杂度会呈指数级增加。马尔可夫假设指一个词的出现只与前面n个词相关，当n比较小时，n元语法往往不准确。例如在一元语法中，“你走先”和“你先走“概率是一样的。



### 6.2  循环神经网络

循环神经网络并非刚性的记忆所有固定长度的序列，而是通过隐藏状态来存储之前时间步的信息。

#### 6.2.1  不含隐藏状态的神经网络

Ht=φ(Xt * Wxh + bh)



#### 6.2.2  含隐藏状态的神经网络

与上节不同，隐藏状态引入一个新的权重参数Whh

Ht=φ(Xt * Wxh + Ht-1 * Whh + bh)

由上式中相邻时间步的隐藏变量Ht和Ht-1可知，这里的隐藏变量能够捕捉至当前时间步序列的历史信息，就像神经网络的记忆一样。所以上式的计算是循环的，使用循环计算的网络即循环神经网络。

![image-20200723142406148](C:\Users\woo\AppData\Roaming\Typora\typora-user-images\6.1.png)



### 6.3  语言数据模型

#### 6.3.1  读取数据集

首先读取数据集，看看前40个字符是什么样的

```
from mxnet import nd
import random
import zipfile

with zipfile.ZipFile('data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars=f.read().decode('utf-8')
corpus_chars[:40]
```

输出：

```
'想要有直升机\n想要和你飞到宇宙去\n想要和你融化在一起\n融化在宇宙里\n我每天每天每'
```

#### 6.3.2  建立字符索引

为了得到索引，我们将数据集里所有不同的字符取出来，然后将其逐一映射到索引来构造词典。

```
#将集合转为list
idx_to_char = list(set(corpus_chars))
#生成字典
char_to_idx=dict([(char,i) for i,char in enumerate(idx_to_char)])
#字典长度
vocab_size=len(char_to_idx)
vocab_size
#查询原始数据集字符在字典中的索引
corpus_indices=[char_to_idx[char] for char in corpus_chars]
sample=corpus_indices[:20]
sample
```

#### 6.3.3  时序数据的采样

与之前章节的数据不同的是，时序数据的一个样本通常包括连续的字符，假设时间步为5，一个样本有5个字符。例：“想” “要” “有” “直”  “升”。我们有两种方式对时序数据采样，分别是随机采样和相邻采样。

#### 6.4  循环神经网络的实现

首先先读取周杰伦专辑歌词数据集

```
import d2lzh as d2l
import math
from mxnet import nd,autograd
from mxnet.gluon import loss as gloss
import time

(corpus_indices,char_to_idx,idx_to_char,vocab_size)=d2l.load_data_jay_lyrics()
```

#### 6.4.1  one_hot向量

假设字符的数量为N，每个字符已经同一个从0到N-1的连续整数值索引一一对应。如果一个字符的索引是整数i，那么就用一个全0的长为N的向量，并将其位置i的值设为1。

```
nd.one_hot(nd.array([0,2]),vocab_size)
```

输出

```
[[1. 0. 0. ... 0. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]]
<NDArray 2x1027 @cpu(0)>
```

每次采样的小批量的形状是（批量大小，时间步数），通过下面这个函数转为可以进入网络的形状（批量大小，词典大小）的矩阵，矩阵个数等于时间步数。

```
def to_onehot(X,size):
    return [nd.one_hot(x,size) for x in X.T]#T是转置

X=nd.arange(10).reshape((2,5))
inputs=to_onehot(X,vocab_size)
X,inputs
```

#### 6.4.2  初始化模型参数

```
num_inputs,num_hiddens,num_outputs=vocab_size,256,vocab_size
ctx=d2l.try_gpu()
print('will use',ctx)

def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01,shape=shape,ctx=ctx)
    #隐藏层参数
    W_xh=_one((num_inputs,num_hiddens))
    W_hh=_one((num_hiddens,num_hiddens))
    b_h=nd.zeros(num_hiddens,ctx=ctx)
    #输出层参数
    W_hq=_one((num_hiddens,num_outputs))
    b_q=nd.zeros(num_outputs,ctx=ctx)
    #附上梯度
    params=[W_xh,W_hh,b_h,W_hq,b_q]
    for param in params:
        param.attach_grad()
    return params
```

#### 6.4.3  定义模型

```
#返回一个形状为（批量大小，隐藏单元个数）的值为0的NDarray元组
def init_rnn_state(batch_size,bun_hiddens,ctx):
    return (nd.zeros(shape=(batch_size,num_hiddens),ctx=ctx),)

#计算在一个时间步里如何计算隐藏状态和输出
def rnn(inputs,state,params):
    W_xh,W_hh,b_h,W_hq,b_q=params
    H,=state
    outputs=[]
    for X in inputs:
        H=nd.tanh(nd.dot(X,W_xh)+nd.dot(H,W_hh)+b_h)
        Y=nd.dot(H,W_hq)+b_q
        outputs.append(Y)
    return outputs,(H,)
```

#### 6.4.4 定义预测函数

```
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx): 
    state = init_rnn_state(1, num_hiddens, ctx) 
    output = [char_to_idx[prefix[0]]] 
    for t in range(num_chars + len(prefix) - 1): 
        # 将上⼀时间步的输出作为当前时间步的输⼊ 
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size) 
        # 计算输出和更新隐藏状态 
        (Y, state) = rnn(X, state, params) 
        # 下⼀个时间步的输⼊是prefix⾥的字符或者当前的最佳预测字符 
        if t < len(prefix) - 1: 
            output.append(char_to_idx[prefix[t + 1]]) 
        else: 
            output.append(int(Y[0].argmax(axis=1).asscalar())) 
    return ''.join([idx_to_char[i] for i in output]) 
```

#### 6.4.5  裁剪梯度

循环神经网络中容易出现梯度衰减或者梯度爆炸，所以用裁剪梯度应对上述情况。

```
def grad_clipping(params, theta, ctx): 
    norm = nd.array([0], ctx) 
    for param in params: 
        norm += (param.grad ** 2).sum() 
    norm = norm.sqrt().asscalar() 
    if norm > theta: 
        for param in params: 
            param.grad[:] *= theta / norm 
```

#### 6.4.6  困惑度

通常使用困惑度来评价语言模型的好坏。



### 6.5  循环神经网络的简单实现

同样的，还是先获取数据集

```
import d2lzh as d2l
import math
from mxnet import nd,autograd
from mxnet.gluon import loss as gloss
import time

(corpus_indices,char_to_idx,idx_to_char,vocab_size)=d2l.load_data_jay_lyrics()
```

#### 6.5.1  定义模型



```
num_hiddens = 256 
#隐藏单元个数为256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
#批量大小为2
batch_size = 2
#初始化隐藏状态
state = rnn_layer.begin_state(batch_size=batch_size)
state[0].shape,len(state)

num_steps=35
X=nd.random.uniform(shape=(num_steps,batch_size,vocab_size))
Y,state_new=rnn_layer(X,state)
Y.shape,len(state_new),state_new[0].shape
```

下面定义一个完整的循环神经网络

```
class RNNModel(nn.Block):
    def __init__(self,rnn_layer,vocab_size,**kwargs):
        super(RNNModel, self).__init__(**kwargs) 
        self.rnn = rnn_layer 
        self.vocab_size = vocab_size 
        self.dense = nn.Dense(vocab_size)
    
    def forward(self, inputs, state): 
        # 将输⼊转置成(num_steps, batch_size)后获取one-hot向量表⽰ 
        X = nd.one_hot(inputs.T, self.vocab_size) 
        Y, state = self.rnn(X, state) 
        # 全连接层会⾸先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出 
        # 形状为(num_steps * batch_size, vocab_size) 
        output = self.dense(Y.reshape((-1, Y.shape[-1]))) 
        return output, state
    
    def begin_state(self, *args, **kwargs): 
        return self.rnn.begin_state(*args, **kwargs) 
```

#### 6.5.2  训练模型

下面定义一个预测函数

```
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char, char_to_idx): 
    # 使⽤RNNModel的成员函数来初始化隐藏状态 
    state = model.begin_state(batch_size=1, ctx=ctx) 
    output = [char_to_idx[prefix[0]]] 
    for t in range(num_chars + len(prefix) - 1): 
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1)) 
        (Y, state) = model(X, state) # 前向计算不需要传⼊模型参数 
        if t < len(prefix) - 1: 
            output.append(char_to_idx[prefix[t + 1]]) 
        else: 
            output.append(int(Y.argmax(axis=1).asscalar())) 
    return ''.join([idx_to_char[i] for i in output]) 
```

训练函数：

```
def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, 
                                num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes): 
    loss = gloss.SoftmaxCrossEntropyLoss() 
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01)) 
    trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0, 'wd': 0})
    
    for epoch in range(num_epochs): 
        l_sum, n, start = 0.0, 0, time.time() 
        data_iter = d2l.data_iter_consecutive( 
            corpus_indices, batch_size, num_steps, ctx) 
        state = model.begin_state(batch_size=batch_size, ctx=ctx) 
        for X, Y in data_iter: 
            for s in state: 
                s.detach() 
            with autograd.record(): 
                (output, state) = model(X, state) 
                y = Y.T.reshape((-1,)) 
                l = loss(output, y).mean() 
            l.backward() 
            # 梯度裁剪 
            params = [p.data() for p in model.collect_params().values()] 
            d2l.grad_clipping(params, clipping_theta, ctx) 
            trainer.step(1) # 因为已经误差取过均值，梯度不⽤再做平均 
            l_sum += l.asscalar() * y.size 
            n += y.size
            
        if (epoch + 1) % pred_period == 0: 
            print('epoch %d, perplexity %f, time %.2f sec' % ( 
                epoch + 1, math.exp(l_sum / n), time.time() - start)) 
            for prefix in prefixes: 
                print(' -', predict_rnn_gluon( prefix, pred_len, model, vocab_size, ctx, idx_to_char, char_to_idx)) 
```

最后，训练数据

```
num_epochs, batch_size, lr, clipping_theta = 250, 32, 1e2, 1e-2 
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开'] 
train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx, corpus_indices, idx_to_char, char_to_idx, 
                            num_epochs, num_steps, lr, clipping_theta, batch_size, pred_period, pred_len, prefixes) 
```



### 6.6  通过时间反向传播

通过时间反向传播是反向传播在循环神经网络中的具体应用，当世界步数较大或者时间步较小时，梯度较容易出现衰减或爆炸。

### 6.7  门控循环单元GRU

更好的捕捉时间序列中时间步距离较大的依赖关系，修改了隐藏状态的计算方式

### 

























