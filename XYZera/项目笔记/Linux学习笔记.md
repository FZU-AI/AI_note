# **Linux学习笔记**



## **Linux内核版与发行版**

> linux内核是一种开放源码的操作系统，由Linux Torvalds负责维护，提供硬件抽象层、硬盘及文件系统控制及多任务功能的系统核心程序。
>
> linux发行版基于linux内核源码，将Linux系统的内核与外围实用程序(Utilities)软件和文档包装起来，并提供一些系统安装界面和系统配置、设定与管理工具，就构成了一种发行版本(distribution)，Linux的发行版本其实就是Linux核心再加上外围的实用程序组成的一个大软件包。
>
> **本文使用的Ubutun系统就是Linux发行版中的一种，其他热门的发行版本还有Red Hat Linux、CentOS等**



## Linux基本命令

| 命令                | 英语                 | 作用                               |
| ------------------- | -------------------- | ---------------------------------- |
| ls                  | list                 | 查看当前文件夹的内容               |
| cd                  | change directory     | 切换文件夹                         |
| rm                  | remove               | 删除指定的文件                     |
| mkdir               | make directory       | 创建目录                           |
| clear               | clear                | 终端清屏                           |
| pwd                 | print work directory | 查看当前所在文件夹                 |
| touch               | touch                | 若文件不存在，则新建文件           |
| tree                | tree                 | 以树状图列出文件目录               |
| cp  源文件 目标文件 | copy                 | 复制文件或者目录                   |
| mv 源文件 目标文件  | move                 | 移动文件或者目录，还可以用于更名   |
| cat                 | concatenate          | 查看文件内容，创建、合并和追加文件 |
| more                | more                 | 分屏显示文件内容                   |
| grep                | grep                 | 搜索文本文件内容                   |



## Linux终端命令格式

> ### **command  [-options]  [parameter]**

> ```python
> - command： 命令名，例如  ls
> - options： 命令选项，例如  rm -r test
> - parameter：传入参数，例如 rm test.txt
> - []表示可选，可以添加，也可以不添加，看命令需求
> ```



## Linux基本命令详解

> ```python
> - ls
> 
> 常用选项：-a（显示所有文件，包括隐藏文件）、-l（列表化显示详细文件信息）、-h（配合-l使用）
> 常用通配符号：*（表示零至多个字符）、？（表示一个字符）、[]（表示匹配字符组中任意一个，例如[abc]）
> ```

> ```python
> - mkdir
> 
> 递归创建文件夹：-p ，例如 mkdir -p a/b/c 就是创建了一个三级文件目录
> ```

> ```python
> - rm
>  
> 常用选项：-r（递归删除文件夹）、-f（强制删除）、（rm -rf  *）根目录下删库跑路命令哈哈
> ```

> ```python
> - cp
> 
> 常用选项：-r（递归复制文件夹）、-i (覆盖前提示)
> ```

> ```python
> - cat
> 
> 常用选项：-b(显示非空行的行号)、-n(显示所有行的行号)
> ```

> ```c
> - grep
> 
> 常用选项：-n（显示所匹配的行号）、-v（打印所有不匹配的行）、-i（忽略大小写）
> 模式查找：^(行首匹配，匹配所有以*为开头的行)、$(行尾匹配，匹配所有以*为结尾的行)
> ```

> ```python
> - echo
> 
> 常与重定向搭配使用：>(输出，会覆盖文件)、>>(追加，在文件中末行追加)
> ```



## Linux远程管理命令

> ```python
> - shutdowm
> 
> 作用：关机或者重启
> 格式：shutdowm 选项 时间
> 选项：-r(表示重启命令)、-c(取消关机命令)
> ```

> ```python
> - ifconfig
> 
> 作用：查看/配置网卡信息
> 查看网卡IP地址：ifconfig | grep inet (采用管道过滤)
> ```

> ```python
> - ping
> 
> 作用：检测到目标网络的连接
> 格式：ping ip地址/网址（检测连通性，用ctrl+c结束ping操作）
> ```

> ```python
> - ssh
> 
> 作用：远程操作服务器
> 格式：ssh [-p port] user@remote
> ```

> ```python
> - scp
> 
> 作用：远程拷贝
> 格式：ssh [-P port] 源文件 user@remote：目标文件
> 选项：-r(用于拷贝文件夹 scp -r demo user@remote:Desktop)
> ```

> #### 免密码登录步骤
>
> - 配置公钥：进入.ssh文件夹，执行 ssh-keygen 命令，然后一路回车即可
> - 上传公钥到服务器：执行 ssh-copy-id -p port user@remote 即可，然后就可以免密码登陆了

> #### 配置别名
>
> - ```python
>   在.ssh文件夹下创建一个config文件，然后进行以下配置
>   ```
>
> - ```python
>   Host 别名[这是你自己随便指定的名字]
>      HostName ip地址
>      User 系统用户名
>      Port 22
>   ```



## Linux权限管理命令

> ```python
> - chmod
> 
> 作用：修改用户对文件的权限
> 设置读写格式：chmod +/- rwx 文件名|目录名
> 文件所有权限：chmod -R 数字 文件名|目录（r:4、w:2、x:1）
> ```

> ```python
> - chgrp
> 
> 作用：递归修改文件/目录的所属组
> 格式：chgrp -R 组名 文件|目录名（需要root权限，前面需要加上sudo）
> ```

> ```python
> - chown
> 
> 作用：修改文件/目录拥有者
> 格式：chown 用户名 文件名|目录名（需要root权限，前面需要加上sudo）
> ```



## Linux用户管理命令

> ```python
> - useradd
> 
> 作用：添加新用户
> 格式：useradd -m -g 组名 新建的用户名（需要root权限，前面需要加上sudo）
> ```

> ```python
> - passwd
> 
> 作用：设置用户密码
> 格式：passwd 用户名（需要root权限，前面需要加上sudo）
> ```

> ```python
> - userdel
> 
> 作用：删除用户
> 格式：userdel -r 用户名
> ```

> ```python
> - id
> 
> 作用：查看用户的UID个GID信息
> 格式：id 用户名
> ```

> ```python
> - who
> 
> 作用：当前所登陆的用户列表
> ```

> ```python
> - usermod
> 
> 作用：设置用户的主组/附加组和登陆shell
> 设置主组/附加组格式：usermod -g/G 组 用户名
> 修改用户登陆shell格式：usermod -s /bin/bash 用户名
> ```

> ```python
> - which
> 
> 作用：查看执行命令的所在位置
> ```

> ```python
> su
> 
> 作用：切换用户
> 格式：su - 用户名（不加用户名的话就切换为root用户）
> ```



## Linux系统管理命令

> ```python
> - cal
> 
> 作用：查看日历
> 常用选项：-y（查询一整年的日历）
> ```

> ```python
> - df
> 
> 作用：显示磁盘剩余空间
> 常用选项：-h（给磁盘大小加上单位，便于阅读）
> ```

> ```python
> - du
> 
> 作用：显示目录下各文件大小
> 常用选项：-h（给磁盘大小加上单位，便于阅读）
> ```

> ```python
> - ps
> 
> 作用：查看进程详细状态
> 常用选项：-a（显示终端所有用户的进程）、-u（显示详细信息）、-x（显示没有控制终端的进程，慎用）
> ```

> ```python
> - top
> 
> 作用：显示正在运行的进程并排序
> ```

> ```python
> - kill
> 
> 作用：结束进程
> 格式：kill [-9] 进程代号（其中-9表示强行终止）
> ```



## 其他命令合集

> ```python
> - find
> 
> 作用：查找文件
> 格式：find [路径] -name 部分文件名（例如：find Destop/ -name "*.txt"）
> ```

> ```python
> - ln
> 
> 作用：创建软链接（类似于快捷方法）
> 格式：ln -s 源文件绝对路径地址 链接文件名
> ```

> ```python
> - tar
> 
> 作用：打包/解包
> 打包格式：tar -cvf 打包文件.tar 被打包的文件/路径
> 解包格式：tar -xvf 打包文件.tar
> 常用选项：-c（打包文件）、-x（解包文件）、-v（显示进度）、-f（指定.tar为结尾的文件名称）
> ```

> ```
> - gzip
> 
> 作用：压缩/解压tar文件，扩展名为xx.tar.gz
> 压缩格式：tar -zcvf 打包文件.tar.gz 被打包的文件/路径
> 解压格式：tar -zxvf 打包文件.tar.gz [-C 解压路径]
> ```

> ```python
> - bzip2
> 
> 作用：压缩/解压tar文件，扩展名为xx.tar.bz2
> 压缩格式：tar -jcvf 打包文件.tar.bz2 被打包的文件/路径
> 解压格式：tar -jxvf 打包文件.tar.bz2 [-C 解压路径]
> ```

> ```python
> - apt
> 
> 作用：下载/卸载/更新软件
> 安装格式：sudo apt install 软件包
> 卸载格式：sudo apt remove 软件名
> 更新格式：sudo apt upgrade（一键更新系统内所有可更新软件）
> ```

