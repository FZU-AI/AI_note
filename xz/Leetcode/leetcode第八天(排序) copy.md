**今天完成题目**:1491,922,1356,1122  
1491:去掉最大最小取平均
- so easy

922:按奇偶排序数组
- 遍历一遍,设置两个索引分别存放即可

1356:根据二进制下1的数目排序
```python
# 内置函数bin转化二进制
>>>bin(10)
'0b1010'
>>> bin(20)
'0b10100'
# 通过key指定两个排序,先按1的个数排序,再按数值排序
sorted(arr, key=lambda x: (bin(x).count('1'),x))
# 为list拓展list
arr.extend(sorted(dictionary[i]))
```

1122:数组的相对顺序排序
- arr.remove(i)一次只删除一个元素