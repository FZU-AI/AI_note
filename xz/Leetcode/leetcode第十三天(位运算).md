**今天完成题目**:693,190,268,371,1731,401,405,1684  
693:交替位二进制数
- pass

190:颠倒二进制位
- pass

268:缺失数字
- pass

371,1731:两整数之和
```python
def getSum(self, a, b):
    # 2^32
    MASK = 0x100000000
    # 整型最大值
    MAX_INT = 0x7FFFFFFF # 第一位表示正负,所以整数最大值=2^31-1
    MIN_INT = MAX_INT + 1 # 负数的补码是在反码的基础上+1,负数最小值=-2^31
    #变成补码后,正负数可以直接相加,而后通过符号位判断正负
    while b != 0:
        # 计算进位
        carry = (a & b) << 1 
        # 取余范围限制在 [0, 2^32-1] 范围内
        a = (a ^ b) % MASK
        b = carry % MASK
    return a if a <= MAX_INT else  ~(a^0xFFFFFFFF)
```
![前32位取反,后32位不变](https://raw.githubusercontent.com/Iamfxz/picRepos/master/imgs/20200731122338.png)

401:二进制手表
1. 字符为hour,整数为minute
2. combinations获取所有的组合情况,permutations可以获取所有的排列情况
3. 限制时间范围
4. 转化为字符型
```python
from itertools import combinations, permutations
h_m = ['8','4','2','1',32,16,8,4,2,1]
list(combinations(h_m,num)) #获得组合情况(有序不重复)
list(permutations(h_m,num)) #获得排列情况(无序有重复)
```

405:数字转换为十六进制数
- if num<0: num=0x100000000+num # 负数补码是一种偏移

1684:整数转换
- 同405类似,将负数补码看作偏移,之后再做比较