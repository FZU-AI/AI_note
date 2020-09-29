**今天完成题目**:876，面试02.07

876:[链表的中间结点](https://leetcode-cn.com/problems/middle-of-the-linked-list/)

- 快慢指针



面试02.07：[链表相交](https://leetcode-cn.com/problems/intersection-of-two-linked-lists-lcci/)

```python
        '''
        这个方法比较巧妙。利用双指针同时移动，在到尾部的时候就交换位置
        因为移动距离相同，如果有交点一定会在交点处相交。这个可以在数学上证明
        a:      A ------- C --------- B
        b: E -- A ------- C --------- D
        设交点为 C 则有:
        L(a) = AB + EC
        L(b) = ED + AC

        ED + AC - (AB + EC) = ED + AC - AB - (EA + AC) = ED - AB - EA
        = ED = (EA + AB) = 0
        '''
```



