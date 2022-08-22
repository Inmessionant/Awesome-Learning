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



#### 基本分支合并

```
git merge test1 master  # 将test1上的差异节点合到master
```



#### 基本分支与节点更新

```
git diff bugfix/cooperate origin/bugfix/cooperate  # 本地bugfix/cooperate与远端差异

git fetch origion feature/xxx:feature/xxx  # 拉下来别人分支到feature/xxx，与自己本地分支不合并，可以通过git log验证，git checkout feature/xxx可以切到这个分支

git cherry-pick commitId  # 把commitId内容拉过来，同时生成了新的commitId
```



#### 合并过程中的冲突处理

```
1.处理conflict文件
2.之后按照add、commit、push操作提交
```



#### 撤销操作

```
git reset hard commitId  # 回退到commitId的版本

git push origin master  # 推动到远端服务器，如果失败，可以使用git push -f origin master强制推送到远端
```

