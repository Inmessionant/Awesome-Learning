

------



## ✅Python



#### regex正则



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



#### What the f*ck Python!





------



## ✅PyTorch



#### 默认梯度累积

- 机器显存小，可以变相增大batchsize；
- weight在不同模型之间交互时候有好处；（动手学习深度学习v2）

```py
accumulation_steps = batch_size // opt.batch_size

loss = loss / accumulation_steps
running_loss += loss.item()
loss.backward()

if ((i + 1) % accumulation_steps) == 0:
	optimizer.step()
	scheduler.step()
	optimizer.zero_grad()
```



------

#### PyTorch提速

- **图片解码**：cv2要比Pillow读取图片速度快
- 加速训练**pin_memory=true / work_numbers=x(卡的数量x4) / prefetch_factor=2 / data.to(device,  no_blocking=True)**
- **DALI库**在GPU端完成这部分**数据增强**，而不是**transform**做图片分类任务的数据增强
- OneCycleLR + SGD / AdamW
- `torch.nn.Conv2d(..., bias=False, ...)`
- DP & DDP 
- 不要频繁在CPU和GPU之间转移数据
- **混合精度训练：**`from torch.cuda import amp`使用`FP16`



------

#### Module & Functional

- **nn.Module**实现的layer是由class Layer(nn.Module)定义的特殊类，**会自动提取可学习参数nn.Parameter**；
- **nn.Functional**中的函数更像是**纯函数**，由def function(input)定义，一般只定义一个操作，其无法保存参数；



- **Module**只需定义 __init__和**forward**，而backward的计算由自动求导机制；
- **Function**需要定义三个方法：**init, forward, backward**（需要自己写求导公式） ；



- **对于激活函数和池化层，由于没有可学习参数，一般使用nn.functional完成，其他的有学习参数的部分则使用nn.Module；**
- 但是**Droupout**由于在训练和测试时操作不同，所以**建议使用nn.Module实现**，它能够通过**model.eval**加以区分；



------

#### Sequential & ModuleList

**区别：**

- **nn.Sequential内部实现了forward函数，而nn.ModuleList则没有实现内部forward函数**；
- **nn.Sequential可以使用OrderedDict对每层进行命名**;
- **nn.Sequential里面的模块按照顺序进行排列的**，所以必须确保前一个模块的输出和下一个模块的输入是一致的；而**nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言**；
- **nn.ModuleList，它是一个储存不同 Module，并自动将每个 Module 的 Parameters 添加到网络之中的容器**；



**nn.Sequential**

- nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的；
- nn.Sequential中可以使用OrderedDict来指定每个module的名字，而不是采用默认的命名方式；
- nn.Sequential内部实现了forward函数；

```python
from collections import OrderedDict

class net_seq(nn.Module):
    def __init__(self):
        super(net_seq, self).__init__()
        self.seq = nn.Sequential(OrderedDict([
                        ('conv1', nn.Conv2d(1,20,5)),
                         ('relu1', nn.ReLU()),
                          ('conv2', nn.Conv2d(20,64,5)),
                       ('relu2', nn.ReLU())
                       ]))
    def forward(self, x):
        return self.seq(x)
net_seq = net_seq()
```



**nn.ModuleList**

- **nn.ModuleList，它是一个储存不同 Module，并自动将每个 Module 的 Parameters 添加到网络之中的容器**：你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，方法和 Python 自带的 list 一样，无非是 extend，append 等操作。但不同于一般的 list，加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中，而使用 Python 的 list 添加的卷积层和它们的 parameters 并没有自动注册到我们的网络中；
- nn.ModuleList需要手动实现内部forward函数；

```python
class net_modlist(nn.Module):
    def __init__(self):
        super(net_modlist, self).__init__()
        self.modlist = nn.ModuleList([
                       nn.Conv2d(1, 20, 5),
                       nn.ReLU(),
                        nn.Conv2d(20, 64, 5),
                        nn.ReLU()
                        ])

    def forward(self, x):
        for m in self.modlist:
            x = m(x)
        return x

net_modlist = net_modlist()
```



------

#### DataLoader & Sampler & DataSet 

```python
class DataLoader(object):
	# DataLoader.next的源代码，__next__函数可以看到DataLoader对数据的读取其实就是用了for循环来遍历数据
    def __next__(self):
        if self.num_workers == 0:  
            indices = next(self.sample_iter)  # Sampler
            # collate_fn的作用就是将一个batch的数据进行合并操作。默认的collate_fn是将img和label分别合并成imgs和labels，所以如果你的__getitem__方法只是返回 img, label,那么你可以使用默认的collate_fn方法，但是如果你每次读取的数据有img, box, label等等，那么你就需要自定义collate_fn来将对应的数据合并成一个batch数据，这样方便后续的训练步骤
            batch = self.collate_fn([self.dataset[i] for i in indices]) # Dataset遍历数据，self.dataset[i]=(img, label)
            if self.pin_memory:
                batch = _utils.pin_memory.pin_memory_batch(batch)
            return batch
```



- **一般来说PyTorch中深度学习训练的流程是这样的： 1. 创建Dateset ；2. Dataset传递给DataLoader； 3. DataLoader迭代产生训练数据提供给模型；**
- 假设我们的数据是一组图像，每一张图像对应一个index，那么如果我们要读取数据就只需要对应的index即可，即代码中的`indices`，而选取index的方式有多种，有按顺序的，也有乱序的，所以这个工作需要`Sampler`完成，`DataLoader`和`Sampler`在这里产生关系；
- 我们已经拿到了indices，那么下一步我们只需要根据index对数据进行读取即可了，这时`Dataset`和`DataLoader`产生关系；

```
-------------------------------------
| DataLoader												|				
|																		|							
|			Sampler -----> Indices				|  													
|                       |						|	
|      DataSet -----> Data					|
|												|						|			
------------------------|------------                    
												|s						
                        Training
```



```python
class DataLoader(object):
  # DataLoader 的源代码
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None)
```



DataLoader 的源代码初始化参数里有两种sampler：`sampler`和`batch_sampler`，都默认为`None`；前者的作用是生成一系列的index，而batch_sampler则是将sampler生成的indices打包分组，得到batch的index；

```python
>>>in : list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
>>>out: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
```



Pytorch中已经实现的`Sampler`有如下几种：`SequentialSampler` 	`RandomSampler`	 `WeightedSampler` 	`SubsetRandomSampler`,需要注意的是DataLoader的部分初始化参数之间存在互斥关系，这个你可以通过阅读[源码](https://github.com/pytorch/pytorch/blob/0b868b19063645afed59d6d49aff1e43d1665b88/torch/utils/data/dataloader.py#L157-L182)更深地理解，这里只做总结：

- 如果你自定义了`batch_sampler`,那么`batch_size`, `shuffle`,`sampler`,`drop_last`这些参数都必须使用默认值；
- 如果你自定义了`sampler`，那么`shuffle`需要设置为`False`；
- 如果`sampler`和`batch_sampler`都为`None`,那么`batch_sampler`使用Pytorch已经实现好的`BatchSampler`,而`sampler`分两种情况：
    - 若`shuffle=True`,则`sampler=RandomSampler(dataset)`
    - 若`shuffle=False`,则`sampler=SequentialSampler(dataset)`




如何自定义Sampler和BatchSampler：查看源代码其实可以发现，所有采样器其实都继承自同一个父类，即`Sampler`,其代码定义如下：

```python
class Sampler(object):
    r"""Base class for all Samplers.
    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.
    .. note:: The :meth:`__len__` method isn't strictly required by
              :class:`~torch.utils.data.DataLoader`, but is expected in any
              calculation involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError
		
    def __len__(self):
        return len(self.data_source)
```

- 所以你要做的就是定义好`__iter__(self)`函数，不过要注意的是该函数的返回值需要是可迭代的，例如`SequentialSampler`返回的是`iter(range(len(self.data_source)))`；
- 另外`BatchSampler`与其他Sampler的主要区别是它需要将Sampler作为参数进行打包，进而每次迭代返回以batch size为大小的index列表。也就是说在后面的读取数据过程中使用的都是batch sampler；



Dataset定义方式如下：

```python
class Dataset(object):
	def __init__(self):
		...
		
	def __getitem__(self, index):
		return ...
	
	def __len__(self):
		return ...
```

- 面三个方法是最基本的，其中`__getitem__`是最主要的方法，它规定了如何读取数据。但是它又不同于一般的方法，因为它是python built-in方法，其主要作用是能让该类可以像list一样通过索引值对数据进行访问。假如你定义好了一个dataset，那么你可以直接通过`dataset[0]`来访问第一个数据；



------

#### Model.Eval & Torch.No_Grad

- **两者都在Inference时候使用，但是作用不相同：**
    - model.eval() 负责改变batchnorm、dropout的工作方式，如在eval()模式下，dropout是不工作的；
    - torch.no_grad() 会关闭自动求导引擎，节省显存和eval的时间；
- **只进行Inference时，`model.eval()`是必须使用的，否则会影响结果准确性； 而`torch.no_grad()`并不是强制的，只影响运行效率；**



------

#### nn.Linear 和nn.Embedding

- **对torch.tensor([1])做Embedding，可以拿到embedding权重当中的第1号位置的一行（查表操作），Linear则会把你的输入和权重做一个矩阵乘法得到输出**；

- **输入不同：**Embedding输入数字，Linear输入one hot向量；

- **本质相同，nn.Embedding等价于torch.one_hot+nn.linear（bias为0）：**查表的操作本质还是相当于一个one_hot向量和权重矩阵做了一次矩阵乘法；

- 虽然从运算过程来说，nn.Embedding与nn.Linear几乎相同，但是nn.Embdding的层的参数是是直接对应于词的表征的，这和nn.Linear（one-hot向量没有任何的语义信息）还是有本质区别的；

- 习惯上，我们在模型的第一层使用的是Embedding，模型的后续不会再使用Embedding，而是使用Linear；

    

```python
import torch

embedding = torch.nn.Embedding(3, 4)
print(embedding.weight)
print(embedding(torch.tensor([1])))

###############################################
Parameter containing:
tensor([[ 1.6238, -0.0947,  0.1135,  1.0270],
        [ 0.3348,  0.2148, -0.5463, -1.3829],
        [-0.3593, -1.0826, -1.0345, -1.5916]], requires_grad=True)
tensor([[ 0.3348,  0.2148, -0.5463, -1.3829]], grad_fn=<EmbeddingBackward0>)

# nn.Embedding是用来将一个数字变成一个指定维度的向量的，比如数字1变成一个128维的向量，数字2变成另外一个128维的向量，这些128维的向量是模型真正的输入。不过这128维的向量并不是永恒不变的，这128维的向量会参与模型训练并且得到更新，从而数字1和2会有更好的128维向量的表示；
```



------

#### Seed 引发的可复现性陷阱



PyTorch模型训练中的**两种可复现性**：

```
1. 在完全不改动代码的情况下重复运行，获得相同的准确率曲线； ->  固定所有随机数种子
2. 改动有限的代码，改动部分不影响训练过程的前提下，获得相同的曲线； -> 改动的代码没有影响random()的调用顺序
```



**1 第一种情况，我们只需要固定所有随机数种子就行**

计算机一般会使用混合线性同余法来生成伪随机数序列，在我们每次调用rand()函数时就会执行一次或若干次下面的递推公式：
$$
x_{n+1}=\left(a x_n+c\right) \bmod (m)
$$

- 当 a 、 c 和 m 满足一定条件时，可以近似地认为 x 序列中的每一项符合均匀分布，通过 x/m 我们可以得到0到1之间的随机数；
- 这类算法都有一个特点，就是一旦固定了序列的初始值 $x_0$ ，整个随机数序列也就固定了，这个初始值就被我们称作种子：
    - 我们在程序的起始位置设定好随机数种子，程序单次执行中第 n 次调用到rand()得到的数值将会是固定的；
    - **一旦程序中rand()函数的调用顺序固定**，无论程序重复运行多少遍，结果都将是稳定的；
- 在PyTorch中我们一般使用seed_everything固定随机数种子，它调用尽量放在所有import之后，其他代码之前；

```python
def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)
```



**2 第二种情况，一定要万分确定改动的代码没有影响random()的调用顺序**

**首先要清楚我提到的固定随机数种子对可复现性起作用的前提：rand()函数调用的次序固定。也就是说，假如在某次rand()调用之前我们插入了其他的rand()操作，那这次的结果必然不同！**

```python
>>> import torch
>>> from utils import seed_everything

>>> seed_everything(0)
>>> torch.rand(5)
tensor([0.4963, 0.7682, 0.0885, 0.1320, 0.3074])  #

>>> seed_everything(0)
>>> _ = torch.rand(1)
>>> torch.rand(5)
tensor([0.7682, 0.0885, 0.1320, 0.3074, 0.6341])  # 偏移一位
```



```
问题描述：在固定随机数种子的前提下，你写了一个训练模型的代码，输出了训练的loss和准确率并绘制了图像，突然你想在每轮训练之后再测一下测试准确率，于是小心翼翼地修改了代码，那么问题来了，训练的loss和准确率会和之前一样吗？ 

-> False，每轮训练完再测试准确率会使用for inputs, labels in dataloader，而这个会引入随机函数；
```



模型测试中唯一不确定的就是DataLoader：按照常规设置，训练时一般使用带shuffle的DataLoader，而测试时使用不带shuffle的，我们进行复现：

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from utils import seed_everything

seed_everything(0)
dataset = TensorDataset(torch.rand((10, 3)), torch.rand(10))
dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
print(torch.rand(5))
# tensor([0.5263, 0.2437, 0.5846, 0.0332, 0.1387])

seed_everything(0)
dataset = TensorDataset(torch.rand((10, 3)), torch.rand(10))
dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
for inputs, labels in dataloader:
    pass
print(torch.rand(5))
tensor([0.5846, 0.0332, 0.1387, 0.2422, 0.8155])
```



**阅读Pytorch中DataLoader的源码可以发现：**`for inputs, labels in dataloader`的`in`先调用后面的迭代器`dataloader`中的`iter()`，每次遍历数据集时DataLoader的`iter()`都会返回一个新的生成器，这个生成器底层有一个`_index_sampler`，shuffle设置为True时它使用`Batch(RandomSampler)`随机采样`batchsize`个数据索引，如果为False则使用`Batch(SequentialSampler)`顺序采样；而这个生成器的基类叫做`BaseDataLoaderIter`，在它的初始化函数中唯一调用了一次随机数函数，用以确定全局随机数种子`base_seed`，且`base_seed`仅使用在其子类`_MultiProcessingDataLoaderIter`中，当我们将`DataLoader`的`worker`数量设置为大于0时，将使用多进程的方式加载数据，在这个子类的初始化函数中会新建`n`个进程，然后将`base_seed`作为进程参数传入：

```python
# 1.
class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader) -> None:
        ...
        # 调用了一次随机
        self._base_seed = torch.empty((), dtype=torch.int64).random_(generator=loader.generator).item()
        ...
    
    
# 2.self._base_seed在_MultiProcessingDataLoaderIter中使用      
... 
w = multiprocessing_context.Process(
    target=_utils.worker._worker_loop,
    args=(self._dataset_kind, self._dataset, index_queue,
          self._worker_result_queue, self._workers_done_event,
          self._auto_collation, self._collate_fn, self._drop_last,
          self._base_seed, self._worker_init_fn, i, self._num_workers,
          self._persistent_workers))
w.daemon = True
w.start()
...


# 3.
def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                 num_workers, persistent_workers):
    ...
    seed = base_seed + worker_id # 确定seed
    random.seed(seed)
    torch.manual_seed(seed)
    if HAS_NUMPY:
        np_seed = _generate_state(base_seed, worker_id)
        import numpy as np
        np.random.seed(np_seed)
    ...
```



**按照PyTorch向后兼容的设计理念，这里无论谁继承_BaseDataLoaderIter这个基类，无论子类是否用到base_seed这个种子，随机数函数都是会被调用的，调用关系如下：**

```python
for inputs, labels in DataLoader(...):
    pass
# 2.in操作符会调用如下
DataLoader()
    DataLoader.self.__iter__()
        DataLoader.self._get_iterator()
            _MultiProcessingDataLoaderIter(DataLoader.self)
                _BaseDataLoaderIter(DataLoader.self)
                    _BaseDataLoaderIter.self._base_seed = torch.empty(
                        (), dtype=torch.int64).random_(generator=DataLoader.generator).item()
# 一般来说generator是None，我们不指定，random_没有from和to时，会取数据类型最大范围，这里相当于随机生成一个大整数
```



- 这一问题的解决方案是在每次DataLoader的in操作调用之前都固定一下随机数种子（1.），**但stable()会使训练时每个epoch内部的shuffle规律相同**；
- 之前我们提到shuffle训练集可以减轻模型过拟合，当每个epoch内部第i个batch的内容都对应相同时模型会训不起来，所以，一个简单的技巧，在传入随机数种子的时候加上一个epoch序号（2.）；
- 这时随机数种子的设定和in操作绑定成了类似的原子操作，所有涉及到random()调用的新增代码都不会影响到准确率曲线的复现了；

```python
# 1.
def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader

for inputs, labels in stable(DataLoader(...), seed):
    pass
  
  
# 2.
def stable(dataloader, seed):
    seed_everything(seed)
    return dataloader
  
for epoch in range(MAX_EPOCH):  # training
    for inputs, labels in stable(DataLoader(...), seed + epoch):
        pass
```



------



## ✅Anaconda



#### 深度学习环境搭建

```python
# conda 创建环境
conda create -n name python=3.8

# 使用pytorch官网的pip/conda命令装torch和torchvision
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


# MacOS安装
# MPS acceleration is available on MacOS 12.3+
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```




```python
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



```python
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

```python
git add READ.md # 没有被跟踪，需要add

git commit READ.md -m "first test"   # 单独提交READ.md文件，-m表示信息
git commit --amend  # 增量提交，可以修改提交信息，esc + :wq

git push origin master  # 推到远端git服务器， git remote显示origin，表示远端名为origin，本地名为master
```



#### 批量操作

```python
git add .  # 全部提交

git commit . -am "first test"

git push origin master
```



#### 分支上开发与查看日志

```python
git checkout - b test1  # 新建一个分支，并且继承了commit节点

git add .  # 全部提交

git commit . -am "modified"

git push origin test1  # 本地分支是test1
```



#### 基本分支合并

```python
git merge test1 master  # 将test1上的差异节点合到master
```



#### 基本分支与节点更新

```python
git diff bugfix/cooperate origin/bugfix/cooperate  # 本地bugfix/cooperate与远端差异

git fetch origion feature/xxx:feature/xxx  # 拉下来别人分支到feature/xxx，与自己本地分支不合并，可以通过git log验证，git checkout feature/xxx可以切到这个分支

git cherry-pick commitId  # 把commitId内容拉过来，同时生成了新的commitId
```



#### 合并过程中的冲突处理

```python
1.处理conflict文件
2.之后按照add、commit、push操作提交
```



#### 撤销操作

```python
git reset hard commitId  # 回退到commitId的版本
git push origin master  # 推动到远端服务器，如果失败，可以使用git push -f origin master强制推送到远端

git checkout READ.md  # 可以回退工作区未提交文件到最近节点内容
git checkout .  # 可以回退工作区所有未提交文件到最近节点内容
```



------



## ✅Makefile



https://github.com/seisman/how-to-write-makefile

