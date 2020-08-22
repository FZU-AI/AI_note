**今天完成题目**:1042,1663,706,705,1658    
1042:不邻接植花
- bfs遍历花
```python
from collections import defaultdict

dict1 = defaultdict(int)
dict2 = defaultdict(set)
dict3 = defaultdict(str)
dict4 = defaultdict(list)
# 如果key为空,返回int,set,str,list的默认空值
```

1663:动物收容所
- 双向队列deque
- 出队时候,记得先判断有无队
- 而后通过popleft()

706:设计哈希映射
- 通过数组+取模的方法设计哈希映射
- 哈希元素使用数组的方法
- 哈希表的长度推荐用质数,可以减少冲突
- for index, val in enumerate(list) 可以获取索引和值.
- del list[index] 可以删除指定元素

705:设计哈希集合
- 跟哈希映射一样,但是在哈希元素的实现上,可以使用链表的方法,更方便插入,删除
```python
class MyHashSet:
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.keyRange = 769 # 质数
        self.bucketArray = [Bucket() for i in range(self.keyRange)]

    def _hash(self, key):
        return key % self.keyRange

    def add(self, key: int) -> None:
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].insert(key)

    def remove(self, key: int) -> None:
        bucketIndex = self._hash(key)
        self.bucketArray[bucketIndex].delete(key)

    def contains(self, key: int) -> bool:
        """
        Returns true if this set contains the specified element
        """
        bucketIndex = self._hash(key)
        return self.bucketArray[bucketIndex].exists(key)

        
class Node:
    def __init__(self, value, nextNode=None):
        self.value = value
        self.next = nextNode

class Bucket:# 在头部插入
    '''
    哈希集合的元素
    '''
    def __init__(self):
        self.head = Node(-1)
    
    def exists(self, val):
        cur = self.head.next
        while cur!=None:
            if cur.value==val:
                return True
            cur = cur.next
        return False

    def insert(self, newVal):
        if not self.exists(newVal):
            newNode = Node(newVal, self.head.next)
            self.head.next = newNode
            
    
    def delete(self, val):
        pre = self.head
        cur = self.head.next
        while cur is not None:
            if cur.value == val:
                pre.next = cur.next
                return
            pre = cur
            cur = cur.next

```

1658:三合一
- 设计三个栈
```python
self.tripleStack = [None]*stackSize*3 
self.top = [0, stackSize , stackSize*2] # 栈顶指针,也是下一个数据的存放位置
self.roof = [stackSize, stackSize*2, stackSize*3] # 栈顶指针最大位置
self.bottom = [0, stackSize, stackSize*2] # 栈顶指针最小位置
```
- 注意,peek函数,也需要按照pop的方法判断,只不过是不删元素.