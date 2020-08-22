**今天完成题目**:957,684  
957:由写斜杠划分区域
![题解示意图](https://pic.leetcode-cn.com/c8cb24b20122246c2a526e08258c9ca12a37c67115189a7874fd642deb62963d-image.png)
- 将每个方框设为四个三角形
- 根据斜杠的方向来合并01,23或02,13
- 根据有无右下结点来合并25,38
- 之后并查集的父结点数即为结果
```python
class UnionFindSet(object):
    def __init__(self, nodes):
        '''
        初始化并查集
        '''
        # 记录每个节点的父节点
        self.fatherMap = {}
        # 各集合的数量
        self.setNumMap = {}
        # 初始化, 每个节点自成一派
        for node in nodes:
            self.fatherMap[node] = node
            self.setNumMap[node] = 1

    def findFather(self, node):
        '''
        递归逻辑:返回当前节点的父节点; 
        '''
        father = self.fatherMap[node]
        if (node != father):
            father = self.findFather(father)
        # 路径压缩
        self.fatherMap[node] = father
        return father

    def isSameSet(self, a, b):
        '''
        判断两个节点a和b是否属于同一集合
        '''
        return self.findFather(a) == self.findFather(b)

    def union(self, a, b):
        '''
        合并集合a到集合b中
        '''
        if a is None or b is None:
            return
        aFather=self.findFather(a)
        bFather = self.findFather(b)
        if (aFather != bFather):
            # 获取a,b集合的数量
            aNum=self.setNumMap[aFather]
            bNum=self.setNumMap[bFather]
            # a集合加入b的集合中
            self.fatherMap[aFather]=bFather
            self.setNumMap[bFather]=aNum + bNum
            # 删除aFather对应的人数纪录
            self.setNumMap.pop(aFather)
```

684:冗余连接
- 以点为集合做并查集,初始集合根结点为点自身
- 每条边代表一个并查集union操作(father[p1]=p0,这里记得p0和p1是两个结点的根结点)
- 当一条边的两个结点早已在同一个集合时,即为冗余