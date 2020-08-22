**今天完成题目**:720  
720:字典中最长的单词
```python
class Solution:
    def longestWord(self, words: List[str]) -> str:
        res='' # 结果
        trie=Trie() # 初始化字典树(前缀树)
        for word in words: #插入前缀树 
            trie.insert(word)
        print(trie.root.children['a'].children)
        for word in words:
            if trie.search(word):
                if len(word) > len(res): # 搜索得到,且长度更大
                    res=word
                elif len(word)==len(res) and word < res: # 长度相同,但字典序更小
                    res=word
        return res

class TrieNode:
    def __init__(self):
        self.end=False # 表示是否存在某个结点
        self.children=collections.defaultdict(TrieNode) # 儿子字典,所有键默认为TrieNode

class Trie:
    def __init__(self):
        self.root=TrieNode()

    def insert(self, word: str) -> None:
        node=self.root
        for s in word:
            node=node.children[s] # node.children[s]默认为TrieNode,这里自动新建儿子
        node.end=True

    def search(self, word: str) -> bool:
        node=self.root
        for s in word:
            node=node.children.get(s) # 获得键s,没有则返回None
            if node is None or not node.end: # 找不到该字符串
                return False
        return True
```
- reduce(function, iterable[, initializer])
  - 用传给 reduce 中的函数 function（有两个参数）先对集合中的第 1、2 个元素进行操作，得到的结果再与第三个数据用 function 函数运算