```
mkdir NewFile  # 创建名为NewFile的文件夹
touch READ.md  # 创建READ.md文件
```



#### 本地基本提交推送

```
git status  # 当前状态
git branch  # 查看分支
git log  # 查看提交日志
git log --name-status  # 多了A  READ.md,表示增加了READ.md

git add READ.md # 没有被跟踪，需要add

git commit .  #     全部提交
git commit READ.md -m "first test"   # 单独提交READ.md文件，-m表示信息

git push origin master  # 推到远端git服务器， git remote显示origin，表示远端名为origin，本地名为master
```



#### 分枝上开发与查看日志

```

```

