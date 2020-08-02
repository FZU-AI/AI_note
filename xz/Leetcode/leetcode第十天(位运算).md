**今天完成题目**:1486,1342,1290,461,1569,1696,136,1667  
1486:数组异或运算
- 符号^表示异或

1342:将数字变成0的操作次数
- 简单,pass

1290:二进制链表转整数z
```python
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
# 遍历方法
# while head.next!=None: # 或while head!=None:
# head=head.next
```

461:汉明距离
- 汉明距离=字符不相同的数量

1569:二进制中1的个数
- bin函数可以将int转换成字符

1696:最大数值
- 通过int((abs(a-b)+a+b)/2)即可获得最大数

136:只出现过一次的数
- 通过所有不同数值的两倍-所有数值=只出现过一次的数

1667:配对交换
- int("001111",2)可以转化二进制到十进制
- str无法将list中的字符结合在一起,"".join(list)才可以

