**今天完成题目**:1750,1643
1750:恢复空格
- 通过一个token数组,记录截止当前位置未识别字符数.
- 从末尾开始遍历比较好,不需要回溯(因为最长匹配)
- 遍历时候token+1,默认未识别,如果识别一个串,则更新当前值token[i]=min(token[i],token[i+d[j]])
- d表示匹配串的长度

1643:滑动窗口的最大值
- 利用滑动窗口最大值的定义
- 写一个找一段数组内最大值,及其相对索引的函数
  - ```python 
    def maxNum(nums,start):
    '''
    返回数组内的最大值及其相对索引
    start表示数组在原数组中的起始点
    '''
    l = len(nums)
    max_num = nums[0]
    idx = start
    for i in range(l):
        if nums[i] >= max_num:
            max_num = nums[i]
            idx = start+i
    return max_num,idx
    ```
- 每次更新一个值
  - 判断该值索引是否和原最大值索引在同一个滑动窗口内,若是则直接比较是否更新
  - 若不在滑动窗口内,则从新调用maxNum寻找这一段的最大值.

933:最近的请求次数
- from collections import deque
- 通过双端队列,每个请求进队,并且出队那些不在3000ms范围内的数