**今天完成题目**:999,1160,1002,1703,867    
999:可以被一步捕获的棋子数  
- 先找到目标值
- 而后向上下左右寻找符合条件

1160:拼写单词  
- 利用Counter实现
- Counter相当于一个默认字典,没有存在的则为0

1002:查找常用字符  
- 利用Counter实现
- Counter.elements()可以将计数器展开为原数组的迭代器
- Counter &符号相当于取两Counter中的较小值,|则是较大值

1703:魔术索引
- 逐个遍历num[i]==i

867:转置矩阵
```python
class Solution:
    def transpose(self, A: List[List[int]]) -> List[List[int]]:
        # *A 表示任意多的参数
        # zip(A[0],A[1],...)表示将所有的数组的第i个压缩成元组
        return [list(i) for i in zip(*A)]
```