**今天完成题目**:104,700,590,1636  
104:二叉树的最大深度
- 同前面某题一样,递归遍历左右子树的最大深度即可

700:二叉搜索树中的搜索
- 比较根结点,从而判断递归左子树还是右子树,或者返回结果

590:后序遍历
```python
    # 递归法
    # 改成在函数里面嵌套函数才可以
    # 不知道为什么直接用postorder作为递归函数会出问题,全局变量缓存?
    def postorder(self, root: 'Node') -> List[int]:
        result = []


        def post(root):
            if not root: #结点不存在直接返回
                return
            else: # 存在结点
                if root.children:  # 有孩子,优先遍历孩子
                    for i in root.children:
                        post(i)
                result.append(root.val)
            return result


        post(root)
        return result

    # 迭代法,使用栈模拟树的遍历
    def postorder(self, root: 'Node') -> List[int]:
        if not root:
            return None
        stack_run = [root]
        result = []
        while stack_run:
            node = stack_run.pop()
            result.append(node.val)
            children = node.children
            for child in children:
                if child:
                    stack_run.append(child)
        result.reverse()
        return result
```

1636:二叉搜索树的第k大值
- 一种是根据右根左的中序遍历
- 还有一种是不断删除最大值
  - 当有右子树时,删除右子树中的最右结点,若它有左子树,直接插入在删除的位置
  - 当没有右子树时,删除根结点,若根有左子树,并将根树变成左子树