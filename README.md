

# Awesome-Learning



```
Some resources include Python, C++, Anaconda, PyTorch, Machine Learning, Deep Learning etc.
```



|            |                      |                   |                  |               |                                  |
| :--------: | :------------------: | :---------------: | :--------------: | :-----------: | :------------------------------: |
|  数学基础  |        微积分        |     线性代数      | 概率论与数理统计 |    凸优化     |              信息论              |
| 计算机基础 | 数据结构（LeetCode） | 操作系统（Linux） |    计算机网络    | 数据库（SQL） | 编程语言（Python、C++、PyTorch） |
|   AI算法   |       机器学习       |     深度学习      |       SLAM       |   自动驾驶    |                                  |
|            |       稀疏量化       |   神经网络编译    |                  |               |                                  |



------



## Anaconda



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



## C++



- [x] C++ Primer
- [ ] Effective C++：改善程序与设计的55个具体做法



- [x] C/C++细节：https://www.bilibili.com/video/BV1nV4y177nk



------



## Python



- [ ] Effective Python：编写高质量Python代码的90个有效方法

- [ ] 利用Python进行数据分析

- [ ] 流畅的Python

- [x] [Numpy QuickStart](https://numpy.org/doc/stable/user/quickstart.html)

- [x] [What the fuck Python! ](https://github.com/robertparley/wtfpython-cn)



- [x] Python小技巧：https://www.bilibili.com/video/BV15r4y1L7v9
- [x] Effective Python：编写高质量Python代码的90个有效方法：https://www.bilibili.com/video/BV1HB4y1C7zH



------



## Operating System



- [ ] 陈海波 - 现代操作系统：原理与实现
- [ ] Operating Systems:Three Easy Pieces




- [ ] 蒋炎岩 - 2022 南京大学 “操作系统：设计与实现”：https://www.bilibili.com/video/BV1Cm4y1d7Ur



------



## Machine Learning



- [ ] 李航 - 统计学习方法
- [ ] 百面机器学习
- [ ] Pattern Recognition and Machine Learning
- [ ] Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow



- [ ] 林轩田 - 机器学习基石：https://www.bilibili.com/video/BV1Cx411i7op
- [ ] 林轩田 - 机器学习技法：https://www.bilibili.com/video/BV1ix411i7yp
- [ ] 李宏毅 - 2019 机器学习：https://www.bilibili.com/video/BV1Ht411g7Ef
- [ ] 十分钟机器学习：https://www.bilibili.com/video/BV1No4y1o7ac



------



## Deep Learning



- [x] [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)
- [ ] 百面深度学习
- [ ] Goodfellow Bengio - Deep Learning
- [ ] 邱锡鹏 - 神经网络与深度学习



- [x] 李沐 - 动手学习深度学习（PyTorch）v2：https://www.bilibili.com/video/BV1if4y147hS

- [x] 李宏毅 - 2021 深度学习：https://www.bilibili.com/video/BV1JA411c7VT

- [ ] 纽约大学 - 深度学习课程PyTorch：https://www.bilibili.com/video/BV1Lb4y1X77t

- [ ] 邱锡鹏 - 神经网络与深度学习：https://www.bilibili.com/video/BV13b4y1177W



------

