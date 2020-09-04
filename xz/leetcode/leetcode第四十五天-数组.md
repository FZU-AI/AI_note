**今天完成题目**:1170,1185,1260,448,308,985  
1170:比较字符串最小字母出现频次
- 比较之前将f(words)排序一下,速度更快

1185:一周中的第几天
```python
'''
        import calendar
        # calendar.monthrange(year,month)获取每月第一天的起始星期和总天数
        weekend = (day+calendar.monthrange(year,month)[0])%7
        result = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return result[weekend]
        '''
        import datetime
        result = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday","Sunday"]
        # 先用datetime.date创建简单日期对象,再用weekday()返回对应星期
        weekend = datetime.date(year,month,day).weekday()
        return result[weekend]
```

1260:二维网格迁移
- 遍历时target = (i*n+j - k)%length
- 然后再将target转换成target_i,target_j
- result[i][j] = grid[target_i][target_j]

448:找到所有数组中消失的数字
```python
for i in range(N):
    temp = abs(nums[i])-1
    if nums[temp]>0:
        nums[temp] = -nums[temp]
```
- 利用索引来存储
- 遍历数组,将其内容所对应的索引取负值,表示已有
- 剩余索引为正的,说明没有

308:连续数列
- 当累计和大于0时,可以继续遍历
- 否则则从新计算累计和

985:查询后的偶数和
- 利用初始的偶数和
- 寻找每次变动后的规律而改变