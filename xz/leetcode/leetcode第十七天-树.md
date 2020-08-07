**今天完成题目**:1628,1584,965,669  
1628:二叉树的最近祖先
- 一开始使用了层次遍历,较为麻烦,需要填满所有空缺的地方,超出了时间限制
- 而后改成深度遍历,AC
- 在递归中,可以使用p_list.copy()来浅拷贝一个数据
```python
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        if not root or root == p or root == q: return root # 找到p或q
        left = self.lowestCommonAncestor(root.left, p, q) # pq谁先找到就返回谁
        right = self.lowestCommonAncestor(root.right, p, q) #pq谁先找到就先返回谁
        if not left: return right # 若pq都在右边,那么最先找到right的就是最近公共祖先
        if not right: return left # 若pq都在左边,那么最先找到left的就是最近公共祖先
        return root # 若pq在左右两边都有,那么root就是结果
```

1584:从上到下打印二叉树
- 利用队列的FIFO特性层序遍历二叉树
- 在每一层利用一个for循环实现一层的遍历,并临时存储下一层
- 队列可以是collections包的deque双向队列[pop()和popleft(),append()和appendleft()]
- 也可以是queue包的Queue[put()入队和get()出队].

1627:二叉搜索树的最近公共祖先
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 或p,q在同一侧,p,q某一个先遍历到,root.val==p.val or root.val==q.val
        if not root or root == p or root == q: return root
        if root.val>p.val and root.val>q.val: # p,q在左子树中
            return self.lowestCommonAncestor(root.left,p,q)
        elif root.val<p.val and root.val<q.val: # p,q在右子树中
            return self.lowestCommonAncestor(root.right,p,q)
        else: # p,q在左右子树中,root为结果
            return root 

```

965:单值二叉树
- 通过栈迭代遍历所有数值,只要有一个不同,返回false即可

669:修剪二叉树
- 先找出范围内的树
- 而后将这些数按顺序插入二叉搜索树中(不需要高度平衡)
