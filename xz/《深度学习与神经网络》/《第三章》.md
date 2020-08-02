# 《第三章》线性模型

## 3.1线性判别函数和决策边界
### 3.1.1二分类
在二分类问题中，我们只需要一个线性判别函数𝑓(𝒙; 𝒘) = 𝒘T𝒙+𝑏. 特征空间 ℝ^𝐷 中所有满足 𝑓(𝒙; 𝒘) = 0的点组成一个**分割超平面**（Hyperplane），称为**决策边界**（Decision Boundary）或**决策平面**（Decision Surface）.特征空间中每个样本点到决策平面的**有向距离**（Signed
Distance）为𝛾 =𝑓(𝒙; 𝒘)/‖𝒘‖ .  
![线性可分](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730161255.png)  
理解:存在权重w,使其和所有特征x,标签y组合起来均大于0.  
### 3.1.2多分类
![多分类三种方式](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730162243.png)  
对于(2),需要(C-1)+(C-2)+...+1=C(C-1)/2个判别函数  
对于(3),argmax(f(x))是使得 f(x)取得最大值所对应的变量点x(或x的集合)。arg即argument，此处意为“自变量”。从而可以理解为c的概率最大.此方法在特征空间中不会有不确定的值.  
![多类线性可分](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730162820.png)

## 3.2logistic回归
 𝑔(⋅) 通常称为激活函数（Activation Function）
 逆函数 𝑔<sup>−1</sup>(⋅)也称为联系函数（Link Function）.
 ![logistic函数](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730163706.png)  
 标准 Logistic函数在机器学习中使用得非常广泛，经常用来将一个实数空间的数映射到(0,1)区间.  
 ![logistic对数几率](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730164216.png)  
 Logistic回归可以看作是预测值为 “ 标签的对数几率”的线性回归模型. 因此， Logistic 回归也称为对数几率回归（Logit Regression）
 ### 3.2.1参数学习
 ![交叉熵](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730164946.png)
 因为logistic函数的导数是: 𝜎′(𝑥) = 𝜎(𝑥)(1 − 𝜎(𝑥)),带入y_hat'=y_hat(1-y_hat),下列中的y为真实标签数值.      
 ![参数更新](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730170001.png)

## 3.3softmax回归
Softmax 回归（Softmax Regression）， 也称为多项（Multinomial）或多类（Multi-Class）的 Logistic回归，是 Logistic回归在多分类问题上的推广.  
![softmax普通表示](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730171838.png)
![softmax向量表示](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730171919.png)  
### 3.3.1参数学习
![softmax风险函数](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730172606.png)  
这边定义,只有C个分类,没有不属于所有分类的数.
![softmax关于W的梯度](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730173924.png)
![推导过程](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730175131.png)  
Softmax回归往往需要使用正则化来约束其参数. 因为𝐶 个权重向量是冗余的,也能避免计算 softmax函数时在数值计算上溢出问题.  

## 3.4感知器（Perceptron）
### 3.4.1参数学习
损失函数是:ℒ(𝒘; 𝒙, 𝑦) = max(0, −𝑦𝒘T𝒙).
每次分错一个样本 (𝒙, 𝑦)时，即 𝑦𝒘T𝒙 < 0，就用这个样本来更新权重.
![感知器参数学习算法](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200730175709.png) 
### 3.4.2收敛性
𝛾是一个正的常数.
![感知器收敛性](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200731150808.png)
![感知器缺点](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200731151352.png)