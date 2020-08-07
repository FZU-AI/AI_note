**今天完成题目**:589,108,897,559  
589:N叉树的前序遍历
- 根左右的顺序
- 递归法和栈模拟的迭代法
108:有序数组转化为二叉搜索树
- 高度平衡二叉树指左右两个子结点高度差绝对值不超过1
- 有序数组递归取中位数作为根生成搜索二叉树
897:递增顺序查找二叉树
- 先中序递归,而后生成指定的树
```python
class Solution:
    def increasingBST(self, root):
        def inorder(node):
            if node:
                yield from inorder(node.left) # 生成器
                yield node.val
                yield from inorder(node.right)

        ans = cur = TreeNode(None) #初始化为空
        for v in inorder(root):
            cur.right = TreeNode(v)
            cur = cur.right # 指针
        return ans.right
```
559:N叉树的最大深度  
- 递归获得所有子树的较大深度