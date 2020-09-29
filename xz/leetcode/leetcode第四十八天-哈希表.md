**今天完成题目**:970,219,771,500,1207   
970:强整数
- 简单 

219:计数质数
``` python
class Solution:
    def countPrimes(self, n: int) -> int:
        # 最小的质数是 2
        if n < 2:
            return 0

        isPrime = [1] * n
        isPrime[0] = isPrime[1] = 0   # 0和1不是质数，先排除掉

        # 埃式筛，把不大于根号n的所有质数的倍数剔除
        for i in range(2, int(n ** 0.5) + 1):
            if isPrime[i]:
                # print(((n - 1 - i * i) // i + 1))
                # 从i*i到n,以步幅i,都设为0
                isPrime[i * i:n:i] = [0] * ((n - 1 - i * i) // i + 1)

        return sum(isPrime)
```
- 遍历2到(根号n)+1
- 如果是素数则将范围内，其倍数均置为非素数

771:宝石与石头
- 简单

500:键盘行
- 简单

1207:独一无二的出现次数
- 计算是出现的次数独一无二