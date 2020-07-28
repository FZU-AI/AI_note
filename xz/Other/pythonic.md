```python
import this
"""
The Zen of Python, by Tim Peters

Beautiful is better than ugly.
漂亮比丑好（格式）
Explicit is better than implicit.
显性比隐性好（调用）
Simple is better than complex.
简单比麻烦好
Complex is better than complicated.
麻烦比复杂好
Flat is better than nested.
平面比嵌套好（代码顺序）
Sparse is better than dense.
稀疏比稠密好（向量）
Readability counts.
可读性很重要
Special cases aren't special enough to break the rules.
特殊情况也要在规则之内
Although practicality beats purity.
虽然实用性比纯粹重要
Errors should never pass silently.
错误不应该静默
Unless explicitly silenced.
除非声明让它静默
In the face of ambiguity, refuse the temptation to guess.
模棱两可下，不要让人去猜
There should be one-- and preferably only one --obvious way to do it.
应该只有一个——最好只有一个——明显的理解方式
Although that way may not be obvious at first unless you're Dutch.
虽然这样要求无法一开始实现，除非你是个荷兰人
Now is better than never.
现在开始要求，总比从未要求好
Although never is often better than *right* now.
尽管从来没有什么比现在“正确”更好。
If the implementation is hard to explain, it's a bad idea.
如果这个实现难以解释，这是一个糟糕的计划
If the implementation is easy to explain, it may be a good idea.
如果这个实现简单理解，这是一个很好的计划
Namespaces are one honking great idea -- let's do more of those!
命名空间是一个很棒的主意——让我们做更多些吧！
"""
# 生成器yield，避免内存溢出
def fibonacci():
    num0 = 0
    num1 = 1
    for i in range(10):
        num2 = num0 + num1
        yield num2
        num0 = num1
        num1 = num2

for i in fibonacci():
    print(i)

# for-else简化循环
for cc in ['UK','ID','JP','US']:
    if cc == 'CN':
        break
else:
    print('no CN')

# try-else简化异常
"""
try:
    db.execute("UPDATE table SET xx = WHERE yy = yy")
except DBError:
    db.rollback()
else:
    db.commit()
"""


# with自动管理资源
with open('pythonic.py') as fp:
    for line in fp:
        print(line[:-1])
"""
1.调用open，返回对象obj
2.调用obj.__enter__()，返回并赋值给fp
3.执行with的代码块
4.执行obj.__exit__()
5.如果发生异常，传给obj.__exit__()，返回False异常继续抛出，否则挂起继续运行
"""

# 列表推导与生成器表达式
squares = [ x * x for x in range(10)]
print(squares)
squares = ( x * x for x in range(10))
for i in squares:
    print(i)

# items遍历map
m = {'one':1, 'two':2,'three':3}
for k,v in m.items():
    print(k,v)
```
