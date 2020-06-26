**autograd**

import mxnet.ndarray as nd</br>
import mxnet.autograd as ag</br>

#梯度方向grad(f)标识的是点A变化最快,变化率最大的方向。</br>

x = nd.array([[1,2],[3,4]])#定义求导的变量</br>
x.attach_grad()#申请求导需要的空间，空间用于存放x的导数</br>
with ag.record():</br>
    z = x*x*x*2 #定义求导的函数</br>
z.backward()</br>
print("x.grad:",x.grad)</br>

"""   </br>

       ① 首先定义待求导的变量。如上代码,通过NDArray模块,定义了一个2X2的向量矩阵x;
       ②为变量求导申请空间。如上,调用NDArray矩阵对象x的attach_grad()方法后,会为x申请一份用于存放求导结果的空间Buffer。对象x附带着导数空间Buffer,当对x求导成功后,便将求导结果存储在x的空间Buffer中;
    
       ③定义关于待求导变量的函数f。定义函数f,需要调用autograd模块的record()方法,这个方法返回的结果可以理解为就是我们定义的函数f,需要在with的声明下调用record()方法。如上,z=x*x*x*2,z其实就是我们定义的关于x的函数;
    
       ④求导。求导只需要函数对象f调用backward()方法,封装好的backward()函数就会自动对变量求导,并将求导结果存储在之前申请的Buffer空间中。如上,z.backward()即是对变量x求导。求导结果:z(x)=2*x^3,z'(x)=6*x^2。求导结果如下:
"""</br>
print("-------------------------------------------------")</br>
#对控制流求导</br>
y = nd.array([[3,4]])</br>
print(nd.sum(x))</br>
print(nd.norm(x))</br>
z = nd.array([5])</br>
print(z.asscalar())</br>
"""</br>

sum()方法返回矩阵中的每个元素相加的和;norm()方法返回矩阵的L2范式,即将矩阵中每个元素平方和相加后,开根号返回结果调用asscalar()方法的矩阵只有一个元素,asscalar()方法返回就是这个唯一元素的值,是一个常量。</br>

"""</br>
#对控制流求导与普通的梯度求导过程唯一的区别是,一般需要定义一个方法function。function根据输入自变量x的不同,通过迭代和判断来选取关于x的函数。</br>

def function(a):#根据自变量a,来选取函数f并返回</br>
    b = a</br>
    while nd.norm(b).asscalar() < 1000: #判断条件为:矩阵b的L2范数<1000</br>
        b = b * 2</br>
        print("b:",b)</br>
        if nd.sum(b).asscalar() > 0:#判断条件为:矩阵b的L1范数>0</br>
            c = b</br>
        else:</br>
            c = 100 * b</br>
    return c</br>

a = nd.array([3,4,5])</br>
a.attach_grad()#申请变量a的求导空间</br>
with ag.record():</br>
    c = function(a)</br>
c.backward()</br>
print("a.grad:",a.grad)</br>