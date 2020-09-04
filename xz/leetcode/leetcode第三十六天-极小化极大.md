**今天完成题目**:866,486,375  
866:石子游戏
- 转换思想,有一个总分,自己先手则最大化,敌人先手则最小化
- 最终判断总分是否大于0,若是,则说明自己先手胜利
- 此题答案恒为True,因为(先手可以选择所有的奇数堆或者偶数堆,对手只能选择和你相反的选择)
  - 显然，亚历克斯总是赢得 2 堆时的游戏。 通过一些努力，我们可以获知她总是赢得 4 堆时的游戏。
  - 如果亚历克斯最初获得第一堆，她总是可以拿第三堆。 如果她最初取到第四堆，她总是可以取第二堆。第一 + 第三，第二 + 第四 中的至少一组是更大的，所以她总能获胜。
  - 我们可以将这个想法扩展到 N 堆的情况下。设第一、第三、第五、第七桩是白色的，第二、第四、第六、第八桩是黑色的。 亚历克斯总是可以拿到所有白色桩或所有黑色桩，其中一种颜色具有的石头数量必定大于另一种颜色的。
```python
# 高阶函数
from functools import lru_cache
class Solution:
    def stoneGame(self, piles):
        N = len(piles)
        # last recently unused,缓存,None表示无限制
        @lru_cache(None)
        def dp(i, j):
            # The value of the game [piles[i], piles[i+1], ..., piles[j]].
            if i > j: return 0
            parity = (j - i) % 2 # 判断是否是亚历克斯(先手)
            if parity == 1:  # first player
                return max(piles[i] + dp(i+1,j), piles[j] + dp(i,j-1)) # 越大,越能获胜
            else:
                return min(-piles[i] + dp(i+1,j), -piles[j] + dp(i,j-1)) # 越小,越能减少对手获得的分数

        return dp(0, N - 1) > 0
```

486:预测赢家
- 类似866,但是起始的数量不一定是偶数
- 可以使用866的方法,略加修改即可

375:猜数字大小 二
- 遍历所有的1-n中的i,可以获得所有的耗费,选择最小的耗费
- cost(1,n)=i+max(cost(1,i−1),cost(i+1,n))
- 方法有两种:
  - 方法一,利用lru_cache和函数递归
  - 方法二,利用dp数组(也是cost数组)
```python
# 方法一
from functools import lru_cache
class Solution:
    def getMoneyAmount(self, n: int) -> int:
        # dp = [0,0,1,3,4,6,8,10]
        
        @lru_cache(None)
        def dp(i,j):
            '''
                dp(i,j):从i-j所需要的最少能保证赢的钱数
            '''
            if i>=j:
                return 0
            else:
                num = 0xffffffff
                for x in range(i,j+1): # 假设猜了x
                    temp = max(dp(i,x-1),dp(x+1,j))+x # 左右两边中更花钱的可能(为了保证赢)
                    if num>temp: # 选择更少的(为了用最少的花费保证能赢)
                        num = temp
                return num
        return dp(1,n)
```