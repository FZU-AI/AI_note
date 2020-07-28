 **今天完成题目：** 874，1403，944，392，1005  
 874： 模拟机器人走路
 - 通过python和贪心算法实现，特别耗时，导致了超时
 - 可以通过逐步前进的方法
 - 先使用map将成对的list转换为元组 tuple ，再用set变成集合，因为元组是不可变的。set(map(tuple,obstacles)) 
 - python中set和list性能差距数百倍
 - 直接使用list转化为set会去重元素
 - map(func, itreable...) 会根据提供的函数对指定序列做映射。python内置函数
 - map第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的**新列表**。
 - 通过import inspect; inspect.getsourcefile(func)可以获得源码文件

1403：非递增顺序的最小子序列
- math.ceil(nums)向上取整，floor向下取整，不要乱取整，会失去原来的一些精度。
- 一开始由于用了取整，做错了
- list.sort()速度很快，仅使用于list

944：删列造序  
- 使用列表的方法速度慢于使用字符串的方法
- sorted()，python内置排序函数，对所有可迭代对象有效
  
392：判断子序列
- 注意输入可能为空值，不要再错了啊

1005：K次取反后最大化的数组和
- 列清楚每种情况，然后将其转化为代码即可
- 记得提高专心度，提高解题速度
