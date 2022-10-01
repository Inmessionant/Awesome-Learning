## ✅Python

------





### 迭代器 & 生成器





------



### 闭包





------



### 装饰器





------



### regex正则



**？的几种用法：**

- “?”元字符规定其前导对象必须在目标对象中连续出现零次或一次，如：abc(d)?可匹配abc和abcd；
- 当该字符紧跟在任何一个其他限制符（*,+,?，{n}，{n,}，{n,m}）后面时，匹配模式是非贪婪的，非贪婪模式尽可能少的匹配所搜索的字符串，而默认的贪婪模式则尽可能多的匹配所搜索的字符串。例如，对于字符串“oooo”，“o+?”将匹配单个“o”，而“o+”将匹配所有“o”；源字符串`str=“dxxddxxd”`中，`d\w*?`会匹配 dx,而`d\w*?d`会匹配 dxxd；



| 元字符 | 描述                                                         |
| ------ | ------------------------------------------------------------ |
| .      | 句号匹配任意单个字符除了换行符。                             |
| [ ]    | 字符种类。匹配方括号内的任意字符。                           |
| [^ ]   | 否定的字符种类。匹配除了方括号里的任意字符                   |
| *      | 匹配>=0个重复的在*号之前的字符。                             |
| +      | 匹配>=1个重复的+号前的字符。                                 |
| ?      | 标记?之前的字符为可选.                                       |
| {n,m}  | 匹配num个大括号之前的字符或字符集 (n <= num <= m).           |
| (xyz)  | 字符集，匹配与 xyz 完全相等的字符串.                         |
| \|     | 或运算符，匹配符号前或后的字符.                              |
| \      | 转义字符,用于匹配一些保留的字符 `[ ] ( ) { } . * + ? ^ $ \ |` |
| ^      | 从开始行开始匹配.                                            |
| $      | 从末端开始匹配.                                              |



正则表达式提供一些常用的字符集简写。如下:

| 简写 | 描述                                               |
| ---- | -------------------------------------------------- |
| .    | 除换行符外的所有字符                               |
| \w   | 匹配所有字母数字，等同于 `[a-zA-Z0-9_]`            |
| \W   | 匹配所有非字母数字，即符号，等同于： `[^\w]`       |
| \d   | 匹配数字： `[0-9]`                                 |
| \D   | 匹配非数字： `[^\d]`                               |
| \s   | 匹配所有空格字符，等同于： `[\t\n\f\r\p{Z}]`       |
| \S   | 匹配所有非空格字符： `[^\s]`                       |
| \f   | 匹配一个换页符                                     |
| \n   | 匹配一个换行符                                     |
| \r   | 匹配一个回车符                                     |
| \t   | 匹配一个制表符                                     |
| \v   | 匹配一个垂直制表符                                 |
| \p   | 匹配 CR/LF（等同于 `\r\n`），用来匹配 DOS 行终止符 |



```python
import re

patten = "[a-zA-Z]*(\d+)(\w+)"
str1 = "qwe1234azAA"
res = re.search(patten, str1)
print(res.group(), res.group(0), res.group(1), res.group(2))
# qwe1234azAA qwe1234azAA 1234 azAA
```



**Group用法：**

- 有几个()就有几个group；
- group() = group(0)表示全部匹配结果， group(1)表示第一个匹配块，以此类推；





------



### What the f*ck Python!





------



## ✅PyTorch







------



## ✅Anaconda



```
# conda 创建环境
conda create -n name python=3.8

# 使用pytorch官网的pip/conda命令装torch和torchvision
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


# MacOS安装
# MPS acceleration is available on MacOS 12.3+
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```




```
# 清理缓存
conda clean -a

# 安装requirements里面的版本
conda install --yes --file requirements.txt

# 测试cuda是否可用
import torch
import torchvision
print(torch.cuda.is_available())
print(torch.version.cuda)

# 删除conda环境
conda remove -n name --all

# conda换源记得去掉default，添加pytorch
```



------



## ✅Git



```
mkdir NewFile  # 创建名为NewFile的文件夹
touch READ.md  # 创建READ.md文件

git status  # 当前状态
git branch  # 查看分支
git branch -r # 远端服务器分支
git branch -a # 本地所有分支
git log  # 查看提交日志
git log -5 # 查看最近5个日志
git log --name-status  # 多了A  READ.md,表示增加了READ.md
git diff commitId1 commitId2  # 比较commitId1和commitId2之间差异

git rm a.txt  # 删除
git mv b.txt temp/  # 移动
vi模式下，双击D可以快速删除1行
```



### 本地基本提交推送

```
git add READ.md # 没有被跟踪，需要add

git commit READ.md -m "first test"   # 单独提交READ.md文件，-m表示信息
git commit --amend  # 增量提交，可以修改提交信息，esc + :wq

git push origin master  # 推到远端git服务器， git remote显示origin，表示远端名为origin，本地名为master
```

批量操作

```
git add .  # 全部提交

git commit . -am "first test"

git push origin master
```



### 分支上开发与查看日志

```
git checkout - b test1  # 新建一个分支，并且继承了commit节点

git add .  # 全部提交

git commit . -am "modified"

git push origin test1  # 本地分支是test1
```



### 基本分支合并

```
git merge test1 master  # 将test1上的差异节点合到master
```



### 基本分支与节点更新

```
git diff bugfix/cooperate origin/bugfix/cooperate  # 本地bugfix/cooperate与远端差异

git fetch origion feature/xxx:feature/xxx  # 拉下来别人分支到feature/xxx，与自己本地分支不合并，可以通过git log验证，git checkout feature/xxx可以切到这个分支

git cherry-pick commitId  # 把commitId内容拉过来，同时生成了新的commitId
```



### 合并过程中的冲突处理

```
1.处理conflict文件
2.之后按照add、commit、push操作提交
```



### 撤销操作

```
git reset hard commitId  # 回退到commitId的版本
git push origin master  # 推动到远端服务器，如果失败，可以使用git push -f origin master强制推送到远端

git checkout READ.md  # 可以回退工作区未提交文件到最近节点内容
git checkout .  # 可以回退工作区所有未提交文件到最近节点内容
```



------



## ✅Makefile
