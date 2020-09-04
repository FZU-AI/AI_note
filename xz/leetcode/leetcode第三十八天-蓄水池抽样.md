**今天完成题目**:398
```python
import random

print( random.randint(1,10) )        # 产生 1 到 10 的一个整数型随机数  
print( random.random() )             # 产生 0 到 1 之间的随机浮点数
print( random.uniform(1.1,5.4) )     # 产生  1.1 到 5.4 之间的随机浮点数，区间可以不是整数
print( random.choice('tomorrow') )   # 从序列中随机选取一个元素
print( random.randrange(1,100,2) )   # 生成从1到100的间隔为2的随机整数

a=[1,3,5,6,7]                # 将序列a中的元素顺序打乱
random.shuffle(a)
print(a)
```

398:随机数索引
- 相当于蓄水池样本为1,总数为count(即target的索引数量)
- 利用random.randint(0,count) < 1时,来更新index(这里1,是蓄水池的样本数)
  - 当有三个数时候
  - 第三个数被选中的概率 = 1/3,此时count在更新之前为2
  - 第二个数被保留的概率 = (1-1/3)*1/2 = 1/3
  - 第一个数被保留的概率 = (1-1/3)\*(1-1/2)\*1 = 1/3