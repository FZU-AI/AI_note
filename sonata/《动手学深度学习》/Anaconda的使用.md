**环境激活与退出**

```python
conda activate base  // 激活conda环境 -- base
conda deactivate 环境名
```

**查看环境基本信息**

```python
conda info  // 查看本环境的基本信息

1. platform
2. conda version
3. python version: 当前环境下的python版本
```

**查看本地环境**

```
conda info -e 

# conda environments:
#
base                  *  /Users/sonata/opt/anaconda3
sonata                   /Users/sonata/opt/anaconda3/envs/sonata
```

**创建/删除环境**

```
conda create -n 名称 python=3.7
conda remove -n 名称
```

**安装环境包**

```
conda install ...
```

