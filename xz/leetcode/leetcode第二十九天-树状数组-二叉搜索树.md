**今天完成题目**:307,1038  
307:区域和检索-数组可修改
```python
class NumArray:
    def __init__(self, nums: List[int]):
        '''初始化，总时间 O(n)'''
        self._nums = [0] + nums # 树状数组
        n = len(nums)
        # 初始化
        for i in range(1, n + 1):
            j = i + self.lowbit(i) # 寻找父结点
            if j < n + 1:
                self._nums[j] += self._nums[i] # 父结点的值=子结点的值的和

    def lowbit(self, x: int) -> int:
        '''低位计数:返回最小一位1的值,例如0b0010返回0b10'''
        return x & (-x)

    def update(self, idx: int, val: int):
        '''将原数组idx下标更新为val, 总时间O(log n)'''
        prev = self.sumRange(idx, idx)    # 计算出原来的值
        idx += 1 # 下标从1开始
        val -= prev    # val 是要增加的值,可正可负
        while idx < len(self._nums): #修改自己及其父结点的值
            self._nums[idx] += val
            idx += self.lowbit(idx)

    def _query(self, idx: int) -> int:
        '''计算数组[0, idx)的元素之和'''
        res = 0
        while idx > 0:
            res += self._nums[idx]
            idx -= self.lowbit(idx) # 寻找儿子结点
        return res

    def sumRange(self, i: int, j: int) -> int:
        '''返回数组[begin, end] 的和'''
        return self._query(j+1) - self._query(i)

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# obj.update(i,val)
# param_2 = obj.sumRange(i,j)
```
- [树状数组的理解](https://www.jianshu.com/p/7cd5ad2f449a)

1038:从二叉树到更大和树
- 反向中序遍历记录较大值的和