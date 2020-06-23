# 1.Git本地仓库

## 1.1本地仓库

​       仓库又叫做版本库，英文名 **repository**,我们可以简单的理解成是一个目录，用于存放代码，这个目录里面所有的文件都可以被Git管理起来，每个文件的修改，删除等操作Git都可以跟踪到。

---

Git本地操作的三个区域：

![](http://qcdnj0ubu.bkt.clouddn.com/git%E5%B7%A5%E4%BD%9C%E7%9A%84%E4%B8%89%E4%B8%AA%E5%8C%BA%E5%9F%9F.png)



## 1.2基本操作

**全局配置**

``` base
git config --global user.name "用户名"
git config --global user.email "邮箱地址"
```

设置好用户信息包括用户名和邮箱地址，当进行项目修改的时候就会记录用户名

---

**Git仓库的初始化**

即让Git知道需要他来管理当前这个目录

```base
git init
```

执行后，打开这个项目目录下，点击隐藏目录你会发现多了一个.git文件夹。不能删除，也不能随意更改。

---

**Git常用指令操作**

1.查看当前工作状态：

```base
git status
```

2.将工作区文件添加到缓存区

```base
说明：git add 指令，可以添加一个文件，也可以同时添加多个文件。
语法一：git add 文件名
语法二：git add 文件名1 文件名2 文件名3 ......
语法三：git add .    【添加当前目录到缓存区中】
```

3.提交至仓库

```base
git commit -m "注释内容"
```

如果创建了新的文件，从add开始，重复上述操作即可。

如果修改了已提交的文件内容，再从add开始重新提交就好了。

**这里的提交等操作只是提交到Git本地仓库。**

## 1.3版本回退操作

版本回退分为两个步骤：

**1.首先查看版本，确定需要回到的时刻点**

```base
git log
git log --pretty=oneline
```

**2.回退操作：**

```base
git reset --hard 提交编号
```

---

当回到过去版本后又想返回时：

**1.先查询之前的编号：**

```base
git reflog
```

**2.然后再执行:**

```base
git reset --hard 提交编号
```

# 2. 远程仓库

1.GitHub创建仓库

2.git添加远程仓库

3.使用克隆命令将线上仓库克隆到本地

```base
git clone 线上仓库地址
```

![](http://qcdnj0ubu.bkt.clouddn.com/%E7%BA%BF%E4%B8%8A%E4%BB%93%E5%BA%93%E5%9C%B0%E5%9D%80.png)

将仓库克隆下来后即可在本地仓库做对应的操作，完成对应操作后需要提交到线上仓库

```base
git push
```

注意：

- 首次提交时需要获取权限，输入github账号的用户名和密码

- 可能会遇到**fatal: HttpRequestException encountered**

  原因是Github 禁用了TLS v1.0 and v1.1，必须更新Windows的git凭证管理器，下载安装后即可。

---

与提交对应的就是获取项目最新的内容：

```base
git pull
```

# 3.分支操作

所有的分支组成一个项目。

在版本回退的内容中，每次提交都会有记录，Git把他们串成时间线，形成类似于时间轴的东西，这个时间轴就是一个分支，我们称之为master分支。

分支的相关指令：

1.查看分支

```base
git branch
```

2.创建分支

```base
git branch 分支名
```

3.切换分支

```base
git checkout 分支名
```

4.删除分支

```base
git branch -d 分支名
```

5.合并分支

```base
git merge 被合并的分支
```

