# 第八章

### 8.1  命令式和符号式编程

之前我们所使用的都是命令式编程，例如：

```python
def add(a, b): 
        return a + b
def fancy_func(a, b, c, d): 
    e = add(a, b) 
    f = add(c, d) 
    g = add(e, f)
    return g
```

虽然使⽤命令式编程很⽅便，但它的运⾏可能很慢。⼀⽅⾯，即使fancy_func函数中的add是被 重复调⽤的函数。另⼀⽅⾯，我们需要保存变量e和f的 值直到fancy_func中所有语句执⾏结束。下面我们用符号式编程实现：

```python
def add_str(): 
    return ''' 
def add(a, b): 
    return a + b 
'''

def fancy_func_str(): 
    return ''' 
def fancy_func(a, b, c, d): 
    e = add(a, b) 
    f = add(c, d) 
    g = add(e, f) 
    return g 
'''

def evoke_str(): 
    return add_str() + fancy_func_str() + ''' 
print(fancy_func(1, 2, 3, 4)) 
''' 
prog = evoke_str() 
print(prog) 
#compile() 函数将一个字符串编译为字节代码。
y = compile(prog, '', 'exec') 
#exec 执行储存在字符串或文件中的Python语句
exec(y)
```

以上定义的3个函数都仅以字符串的形式返回计算流程。最后，我们通过compile函数编译完整的 计算流程并运⾏。

#### 8.1.1  混合式编程取两者之长

Gluon通过提供混合式编程的⽅ 式做到了这⼀点， 在混合式编程中，我们可以通过使⽤HybridBlock类或者HybridSequential类构建模型。 默认情况下，它们和Block类或者Sequential类⼀样依据命令式编程的⽅式执⾏。当我们调 ⽤hybridize函数后，Gluon会转换成依据符号式编程的⽅式执⾏。

#### 8.1.2  使⽤HybridSequential类构造模型 

```python
from mxnet import nd, sym 
from mxnet.gluon import nn 
import time

def get_net(): 
    net = nn.HybridSequential() # 这⾥创建HybridSequential实例 
    net.add(nn.Dense(256, activation='relu'), 
            nn.Dense(128, activation='relu'), 
            nn.Dense(2)) 
    net.initialize() 
    return net

x = nd.random.normal(shape=(1, 512)) 
net = get_net() 
#通过调⽤hybridize函数来编译和优化HybridSequential实例中串联的层的计算
net.hybridize()
net(x)
```

下⾯通过⽐较调⽤hybridize函数前后的计算时间来展⽰符号式编程的性能提升：

```python
def benchmark(net, x): 
        start = time.time() 
        for i in range(1000): 
            _ = net(x) 
        nd.waitall() # 等待所有计算完成⽅便计时 
        return time.time() - start
    
net = get_net() 
print('before hybridizing: %.4f sec' % (benchmark(net, x))) 
net.hybridize() 
print('after hybridizing: %.4f sec' % (benchmark(net, x))) 
```

虽然符号式编程会带来性能提升，但是也会降低灵活性。



### 8.2  异步计算

MXNet使用异步计算来提升计算性能，有助于在内存资源有限的情况下主动降低计算性能从而减小内存开销。下面是本节所需要的包：

```python
from mxnet import autograd, gluon, nd 
from mxnet.gluon import loss as gloss, nn 
import os 
import subprocess 
import time 
```

#### 8.2.1  MXNet中的异步计算

MXNet包括⽤⼾直接⽤来交互的前端和系统⽤来执⾏计算的后端，MXNet通过前端线程和后端线程的交互实现异步计算。异步计算指，前端线程⽆须等待当前指令从后端线程返回结果就继续执⾏后⾯的指令（而后端一般是C++）。

例如：

```python
a = nd.ones((1, 2)) 
b = nd.ones((1, 2)) 
c = a * b + 2 
c 
```

在异步计算中，Python前端线程执⾏前3条语句的时候，仅仅是把任务放进后端的队列⾥就返回了。当最后⼀条语句需要打印计算结果时，Python前端线程会等待C++后端线程把变量c的结果计算完。此设计的⼀个好处是，这⾥的Python前端线程不需要做实际计算。因此，⽆论Python的 性能如何，它对整个程序性能的影响很小。只要C++后端⾜够⾼效，那么不管前端编程语⾔性能 如何，MXNet都可以提供⼀致的⾼性能。 

#### 8.2.2  同步

wait_to_read函数、waitall函数、asnumpy函数、asscalar函数和print函数 会触发让前端等待后端计算结果的⾏为，这类函数通常称为同步函数。 此外，任何将NDArray转换成其他不⽀持异步计算的数据结构的操作都会让前端等待计算结果。 

#### 8.2.3  异步原理

前端和后端的交互⼤约可以分为3个阶段： 

> 1. 前端令后端将计算任务y = x + 1放进队列； 
> 2. 后端从队列中获取计算任务并执⾏真正的计算；
> 3. 后端将计算结果返回给前端。 

我们将这3个阶段的耗时分别设为t1,t2,t3。如果不使⽤异步计算，执⾏1000次计算的总耗时⼤约 为1000(t1 +t2 +t3)；如果使⽤异步计算，由于每次循环中前端都⽆须等待后端返回计算结果，执 ⾏1000次计算的总耗时可以降为t1 + 1000t2 + t3（假设1000t2 > 999t1）。 

#### 8.2.4  异步计算对内存的影响

为了解释异步计算对内存使⽤的影响，让我们先回忆⼀下前⾯章节的内容。在前⾯章节中实现的 模型训练过程中，我们通常会在每个小批量上评测⼀下模型，如模型的损失或者精度。细⼼的读 者也许已经发现了，这类评测常⽤到同步函数，如asscalar函数或者asnumpy函数。如果去掉这些同步函数，前端会将⼤量的小批量计算任务在极短的时间内丢给后端，从而可能导致占⽤更多内存。当我们在每个小批量上都使⽤同步函数时，前端在每次迭代时仅会将⼀个小批量的任务 丢给后端执⾏计算，并通常会减小内存占⽤。

 由于深度学习模型通常⽐较⼤，而内存资源通常有限，建议⼤家在训练模型时对每个小批量都使 ⽤同步函数，例如，⽤asscalar函数或者asnumpy函数评价模型的表现。类似地，在使⽤模型 预测时，为了减小内存的占⽤，也建议⼤家对每个小批量预测时都使⽤同步函数，例如，直接打 印出当前小批量的预测结果。 



### 8.3  自动并行计算

MXNet后端会⾃动构建计算图。通过计算图，系统可以知道所有计算的依赖关系，并可以选择将没有依赖关系的多个任务并⾏执⾏来获得计算性能的提升。例如“异步计算”⼀节的第⼀个例⼦⾥依次执⾏了a = nd.ones((1, 2))和b = nd.ones((1, 2))。这两步计算之间并没有依赖关系，因此系统可以选择并⾏执⾏它们。 

通常，⼀个运算符会⽤到所有CPU或单块GPU上全部的计算资源。例如，dot运算符会⽤到所有CPU（即使是⼀台机器上有多个CPU处理器）或单块GPU上所有的线程。如果每个运算符的计算量⾜够⼤，只在CPU上或者单块GPU上并⾏运⾏多个运算符时，每个运算符的运⾏只分到CPU或单块GPU上部分计算资源。即使这些计算可以并⾏，最终计算性能的提升可能也并不明显。本节中探讨的⾃动并⾏计算主要关注同时使⽤CPU和GPU的并⾏计算，以及计算和通信的并⾏。 



### 8.4  多GPU计算

无



