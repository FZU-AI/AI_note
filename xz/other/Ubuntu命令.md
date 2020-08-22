1. git clone + ssh or https # 克隆文件，https可以代理，速度较快  
2. sudo apt-get remove --purge + 软件名称 # 卸载软件   
3. sudo apt-get autoremove --purge 软件名称  # 卸载软件   
4. ubuntu终端下复制粘贴
   - ctrl + shift + c，ctrl + shift + v
   - 直接鼠标左键选中要复制的命令，然后在需要粘贴的地方按一下鼠标滚轮即可
5. gedit+文件 # 编辑文件
6. /boot/grub/grub.cfg # 引导文件的配置,splash quiet nomodeset recovery是四种开机引导的方式
7. 开机卡在logo界面解决办法: 修改开机引导方式quiet splash $vt_handoff 为 acpi_osi=linux nomodest
8. 更新资源包
   1. sudo apt update
   2. sudo apt upgrade
   3. sudo apt autoremove
9. apt search <软件包> 可以模糊搜索所有相关软件
10. 环境变量
    1.  env # 查看所有变量
    2.  export xxx=x # 临时添加环境变量
    3.  echo $http_proxy # 输出某个变量的值
11. 安装proxychains实现终端全局网络代理
    1.  只需要在运行任何命令前加上proxychains即可
    2.  proxychains gnome-terminal # 用全局代理方式打开终端,方便下载
12. rm -rf dirname # 强制删除某个文件夹及其内容,慎用
    1.  rm -r filename # 递归删除子文件内容,-R可以删除文件夹
    2.  rmdir dirname # 删除空目录
    3.  mkdir dirname # 创建文件
13. jupyter nbconvert --to markdown filename # 将ipynb转换为markdown等格式
14. locate 文件名 # 通过数据库定位文件的位置,最快最好
    - whereis 文件名 # 模糊查找
    - sudo find / -name 文件名 # 精确全局查找