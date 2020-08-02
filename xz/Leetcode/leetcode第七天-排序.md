**今天完成题目**:1502,1370,349  
1502:等差数列
- 太简单了,无fuck说
```python
l = [1,2,3]
all(i in l for i in range(1,4)) # 用于判断是否全部非空
# all函数,用于判断给定的可迭代参数 iterable 中的所有元素是否都为 TRUE，如果是返回 True，否则返回 False。元素除了是 0、空、None、False 外都算 True。
```
1370:上升下降字符串
- 字典法:这道题目按着规则写就可以了,通过counter方法可以快速实现列表元素统计.
- 桶计数:设立26个字母26个桶,更容易实现,一定程度上用空间换时间.
```python
from collections import Counter
Counter(s) # 统计list元素个数,返回dictionary
sorted(counter.items(),key=lambda x:x[0]) # 按照key的大小排序dictionary,第一个参数要iterable
boolverbose = not boolverbose # 取反
dictionary = {} # 字典定义
dictionary.append(key) # 字典添加
dictionary.pop(key)  # 字典删除
```
349:交集
- list转换成set,利用&符号取交集即可