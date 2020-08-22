**今日完成题目**:208,677,421  
208:实现前缀树
```python
class TrieNode:
    def __init__(self):
        self.end=False # 表示是否存在某个结点
        self.children=collections.defaultdict(TrieNode) # 儿子字典,所有键默认为TrieNode

class Trie:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()


    def insert(self, word: str) -> None:
        """
        Inserts a word into the trie.
        """
        node = self.root
        for s in word:
            node = node.children[s] # node.children[s]默认为TrieNode,这里自动新建儿子
        node.end = True
        # print("insert " + word)


    def search(self, word: str) -> bool:
        """
        Returns if the word is in the trie.
        """
        node = self.root
        # print("search " + word)
        for s in word:
            node = node.children.get(s) # 获得键s,没有则返回None
            if node is None: # 前缀匹配失败
                return False
        if node.end: # 该字符有结尾,即匹配成功
            return True
        else:
            return False

    def startsWith(self, prefix: str) -> bool:
        """
        Returns if there is any word in the trie that starts with the given prefix.
        """
        node = self.root
        for s in prefix:
            node = node.children.get(s) # 获得键s,没有则返回None
            if node is None: # 前缀匹配失败
                return False
        return True
```

677:键值映射
- 在实现前缀树的方法中,修改END为val,代表有无值,默认为0
- 计算总和时候,历匹配前缀后,bfs遍所有儿子结点的值

421:数组中两个数的最大异或值
- 首先计算数组中最大数的二进制长度 L。
- 初始化 max_xor = 0。
- 从 i = L - 1遍历到 i = 0（代表着从最左侧的比特位 L - 1遍历到最右侧的比特位 00）：
  - 将 max_xor 左移，释放出下一比特位的位置。
  - 初始化 curr_xor = max_xor | 1（即将 max_xor 最右侧的比特置为 1）。
  - 遍历 nums，计算出长度为 L - i 的所有可能的按位前缀。
    - 将长度为 L - i 的按位前缀加入哈希集合 prefixes，按位前缀的计算公式如下：num >> i。
  - 遍历所有可能的按位前缀，检查是否存在 p1，p2 使得 p1^p2 == curr_xor。比较简单的做法是检查每个 p，看 curr_xor^p 是否存在。
    - 如果存在，就将 max_xor 改为 curr_xor（即将 max_xor 最右侧的比特位改为 1）。
    - 如果不存在，max_xor 最右侧的比特位继续保持为 0。
- 返回 max_xor。

```python
class Solution:
    def findMaximumXOR(self, nums: List[int]) -> int:
        L = len(bin(max(nums))) - 2 # 最大数值的二进制长度
        max_xor = 0
        for i in range(L)[::-1]: # 从最大位开始遍历
            # go to the next bit by the left shift
            max_xor <<= 1 
            # curr_xor的最后一位默认为1
            curr_xor = max_xor | 1
            # compute all existing prefixes 
            # of length (L - i) in binary representation
            prefixes = {num >> i for num in nums} # 所有前缀
            # Update max_xor, if two of these prefixes could result in curr_xor.
            # Check if p1^p2 == curr_xor, i.e. p1 == curr_xor^p2
            # print(max_xor,curr_xor, prefixes)
            max_xor |= any(curr_xor^p in prefixes for p in prefixes)
                    
        return max_xor
```