**今天完成题目**:1690,1137,1716  
1690:汉诺塔问题
```python
# 先将n-1个移动到缓冲,再将最大一个移到目标
def hanota(n, A, B, C):
    '''
    将A中n个移到C,以B作为缓冲
    '''
    # 若只有一个,直接移入
    if n == 1:
        C.append(A.pop())
        return
    # 将A中n-1个移到B,以C作为缓冲
    hanota(n-1, A, C, B)
    # 将A中剩余最大一个,移到C
    C.append(A.pop())
    # 将B中n-1个移到C,以A作为缓冲
    hanota(n-1, B, A, C)
hanota(len(A), A, B, C)
```

1137:第 N 个泰波那契数
- return tribonacci(n-1)+tribonacci(n-2)+tribonacci(n-3)
- 可以用查表法,将结果记录下来

1716:跳水板
- 用数学的方法,计算出长短板的差值diff,每次加上差值即为一个结果

1577:青蛙跳水台
- 发现数字中的规律,本质上类似斐波那契数列,只不过是{1,1,2,3.....}

1576:斐波那契数列
- 使用查表法,发现字典不一定比list高效.