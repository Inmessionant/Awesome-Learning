```
mkdir NewFile  # 创建名为NewFile的文件夹
touch READ.md  # 创建READ.md文件

git status  # 当前状态
git branch  # 查看分支
git log  # 查看提交日志
git log -5 # 查看最近5个日志
git log --name-status  # 多了A  READ.md,表示增加了READ.md

git rm a.txt  # 删除
git mv b.txt temp/  # 移动
```



#### 本地基本提交推送



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



#### 分支上开发与查看日志

```
git checkout - b test1  # 新建一个分支，并且继承了commit节点

git add .  # 全部提交

git commit . -am "modified"

git push origin test1  # 本地分支是test1
```

